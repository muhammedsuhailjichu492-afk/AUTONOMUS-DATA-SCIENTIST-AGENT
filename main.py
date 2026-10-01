

import argparse
import sys
from agent_core.orchestrator import AutonomousDataScientistAgent


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Scientist Agent — end-to-end data science pipeline."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--target", required=True, help="Name of the target column to predict.")
    parser.add_argument("--objective", default="", help="Free-text business objective / context.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write plots and report to.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM-assisted decisions/narrative, even if ANTHROPIC_API_KEY is set.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key. Falls back to the ANTHROPIC_API_KEY environment variable.",
    )
    parser.add_argument(
        "--llm-model", default="claude-sonnet-4-6",
        help="Anthropic model ID to use for LLM-assisted features (default: claude-sonnet-4-6).",
    )
    args = parser.parse_args()

    agent = AutonomousDataScientistAgent(
        output_dir=args.output_dir,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        use_llm=not args.no_llm,
        api_key=args.api_key,
        llm_model=args.llm_model,
    )

    try:
        result = agent.run(
            csv_path=args.csv,
            target_col=args.target,
            business_objective=args.objective,
        )
    except Exception as e:
        print(f"[Agent] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== RUN COMPLETE ===")
    print(f"Best model : {result.best_model_name}")
    print(f"Metrics    : {result.metrics}")
    print(f"Report     : {result.report_path}")
    print(f"All outputs: {result.output_dir}/")
    print(f"LLM assist : {result.llm_status}")


if __name__ == "__main__":
    main()
