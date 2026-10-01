

from __future__ import annotations
import os
from typing import Optional

import pandas as pd
from dataclasses import dataclass, field

from .profiler import DataProfiler, DataProfile
from .cleaner import DataCleaner
from .eda import EDAEngine
from .feature_engineer import FeatureEngineer
from .model_selector import ModelSelector
from .trainer import Trainer
from .evaluator import Evaluator
from .reporter import ReportGenerator
from .llm_client import LLMClient, DEFAULT_MODEL
from .llm_advisor import LLMAdvisor, CleaningAdvice, ModelAdvice
from .narrator import Narrator
from .chat import DataChatAgent, build_context_md


@dataclass
class AgentRunResult:
    profile: DataProfile
    cleaned_df: pd.DataFrame
    best_model_name: str
    best_pipeline: object
    metrics: dict
    report_path: str
    output_dir: str
    llm_enabled: bool = False
    llm_status: str = "disabled"  # "enabled", "disabled", or error message
    cleaning_advice: Optional[CleaningAdvice] = None
    model_advice: Optional[ModelAdvice] = None
    chat_agent: Optional[DataChatAgent] = field(default=None, repr=False)
    eda_plots: list[str] = field(default_factory=list, repr=False)
    evaluation_plot: Optional[str] = None


class AutonomousDataScientistAgent:
    """
    Usage:
        agent = AutonomousDataScientistAgent(output_dir="outputs/run1", use_llm=True)
        result = agent.run(
            csv_path="data.csv",
            target_col="churn",
            business_objective="Predict customer churn to prioritize retention outreach.",
        )
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_llm: bool = True,
        api_key: str | None = None,
        llm_model: str = "claude-sonnet-4-6",
    ):
        self.output_dir = output_dir
        self.test_size = test_size
        self.cv_folds = cv_folds
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize LLM components if enabled
        self.use_llm = use_llm
        self.llm_status = "disabled"
        self.llm_client = LLMClient(api_key=api_key, model=llm_model, enabled=use_llm)
        self.llm_advisor = None
        self.narrator = None

        if self.llm_client.available:
            self.llm_advisor = LLMAdvisor(self.llm_client)
            self.narrator = Narrator(self.llm_client)
            self.llm_status = f"enabled ({llm_model})"
            print(f"[Agent] LLM enabled: {self.llm_status}")
        else:
            self.llm_status = self.llm_client.status_message()
            print(f"[Agent] LLM {self.llm_status}")

        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.eda_engine = EDAEngine(output_dir=self.output_dir)
        self.feature_engineer = FeatureEngineer(test_size=test_size)
        self.model_selector = ModelSelector()
        self.evaluator = None  # created after task_type is known
        self.reporter = ReportGenerator(output_dir=self.output_dir)

    def run(self, csv_path: str, target_col: str, business_objective: str = "") -> AgentRunResult:
        dataset_name = os.path.basename(csv_path)
        print(f"[Agent] Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # ---- Stage 1: Data Profiling ----
        print("[Agent] Stage 1/8: Data Profiling")
        profile = self.profiler.profile(df, target_col=target_col)
        if profile.task_type is None:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        print(f"[Agent]   -> Detected task type: {profile.task_type}")

        # ---- Stage 2: Data Cleaning ----
        print("[Agent] Stage 2/8: Data Cleaning")
        # Get LLM advice on cleaning strategy if available
        cleaning_advice = None
        if self.llm_advisor:
            try:
                cleaning_advice = self.llm_advisor.advise_cleaning(
                    df=df,
                    profile=profile,
                    business_objective=business_objective,
                )
                if cleaning_advice and cleaning_advice.source == "llm":
                    print(f"[Agent]   -> LLM cleaning advice: {cleaning_advice.rationale}")
            except Exception as e:
                print(f"[Agent]   -> LLM cleaning advice failed: {e}")

        cleaned_df = self.cleaner.clean(df, profile, advice=cleaning_advice)
        # re-profile the cleaned data so EDA/feature stages see accurate types
        clean_profile = self.profiler.profile(cleaned_df, target_col=target_col)
        clean_profile.task_type = profile.task_type  # preserve original decision

        # ---- Stage 3: EDA ----
        print("[Agent] Stage 3/8: Exploratory Data Analysis")
        eda_plots = self.eda_engine.run(cleaned_df, clean_profile)

        # ---- Stage 4: Feature Engineering ----
        print("[Agent] Stage 4/8: Feature Engineering")
        preprocessor = self.feature_engineer.build_preprocessor(cleaned_df, clean_profile)
        X_train, X_test, y_train, y_test = self.feature_engineer.split(cleaned_df, clean_profile)

        # ---- Stage 5: Model Selection ----
        print("[Agent] Stage 5/8: Model Selection")
        candidates = self.model_selector.get_candidates(
            task_type=profile.task_type, n_rows=len(cleaned_df), n_features=cleaned_df.shape[1]
        )
        print(f"[Agent]   -> Candidate models: {list(candidates.keys())}")

        # Get LLM advice on model selection if available
        if self.llm_advisor:
            try:
                model_advice = self.llm_advisor.advise_models(
                    task_type=profile.task_type,
                    candidate_models=list(candidates.keys()),
                    n_rows=len(cleaned_df),
                    n_features=cleaned_df.shape[1],
                    business_objective=business_objective,
                )
                if model_advice and model_advice.source == "llm" and model_advice.selected_models:
                    candidates = {m: candidates[m] for m in model_advice.selected_models if m in candidates}
                    print(f"[Agent]   -> LLM-selected models: {list(candidates.keys())}")
                    if model_advice.rationale:
                        print(f"[Agent]   -> Rationale: {model_advice.rationale}")
            except Exception as e:
                print(f"[Agent]   -> LLM model advice failed: {e}")

        # ---- Stage 6: Training ----
        print("[Agent] Stage 6/8: Training & Cross-Validation")
        trainer = Trainer(preprocessor=preprocessor, task_type=profile.task_type, cv_folds=self.cv_folds)
        trainer.train_all(candidates, X_train, y_train)
        best = trainer.best()
        print(f"[Agent]   -> Best model: {best.name} (CV score {best.cv_mean:.4f})")

        # ---- Stage 7: Evaluation ----
        print("[Agent] Stage 7/8: Evaluation")
        self.evaluator = Evaluator(output_dir=self.output_dir, task_type=profile.task_type)
        class_names = None
        if self.feature_engineer.label_encoder is not None:
            class_names = list(self.feature_engineer.label_encoder.classes_)
        metrics = self.evaluator.evaluate(best.pipeline, X_test, y_test, class_names=class_names)
        print(f"[Agent]   -> Metrics: {metrics}")

        # ---- Stage 8: Report ----
        print("[Agent] Stage 8/8: Report Generation")
        
        # Generate summary using LLM if available
        summary = None
        if self.narrator:
            try:
                summary = self.narrator.narrate(
                    dataset_name=dataset_name,
                    business_objective=business_objective or "Not specified.",
                    profile_md=clean_profile.to_markdown(),
                    cleaning_log=self.cleaner.log,
                    feature_log=self.feature_engineer.log,
                    leaderboard_md=trainer.leaderboard_markdown(),
                    best_model_name=best.name,
                    evaluation_md=self.evaluator.metrics_markdown(),
                )
                if summary:
                    print(f"[Agent]   -> LLM narrative generated")
            except Exception as e:
                print(f"[Agent]   -> LLM narrative failed: {e}")

        report_path = self.reporter.generate(
            dataset_name=dataset_name,
            business_objective=business_objective or "Not specified.",
            profile_md=clean_profile.to_markdown(),
            cleaning_log=self.cleaner.log,
            feature_log=self.feature_engineer.log,
            eda_plots=eda_plots,
            leaderboard_md=trainer.leaderboard_markdown(),
            best_model_name=best.name,
            evaluation_md=self.evaluator.metrics_markdown(),
            evaluation_plot=self.evaluator.plot_path,
            summary=summary,
        )
        print(f"[Agent] Done. Report saved to: {report_path}")

        context_md = build_context_md(
            dataset_name=dataset_name,
            business_objective=business_objective,
            profile_md=clean_profile.to_markdown(),
            cleaning_log=self.cleaner.log,
            feature_log=self.feature_engineer.log,
            leaderboard_md=trainer.leaderboard_markdown(),
            best_model_name=best.name,
            evaluation_md=self.evaluator.metrics_markdown(),
        )
        chat_agent = DataChatAgent(self.llm_client, context_md)

        return AgentRunResult(
            profile=clean_profile,
            cleaned_df=cleaned_df,
            best_model_name=best.name,
            best_pipeline=best.pipeline,
            metrics=metrics,
            report_path=report_path,
            output_dir=self.output_dir,
            llm_enabled=self.llm_client.available,
            llm_status=self.llm_status,
            chat_agent=chat_agent,
        )
