import os.path

from pipeline import Pipeline
from argparse import ArgumentParser, Namespace
import logging

logger = logging.getLogger(__name__)


def add_args(parser: ArgumentParser):
    parser.add_argument("--config", required=True, help="pipline configuration")
    parser.add_argument(
        "--user_src", default='../solvers/tutorial_solvers', type=str,
        help="provide user src root with str: './src' or list of str: ['../solvers/s1','../solvers/s2']"
    )
    parser.add_argument(
        "--input", default=None, type=str,
        help="A string with LLM response or a file contains multiple lines of response"
    )
    parser.add_argument("--output", default=None, type=str, help="Output path")


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    input_content = getattr(args, "input", None)
    pipeline = Pipeline(args)
    if os.path.isfile(input_content):
        with open(input_content, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                content = line.strip()
                if len(content) > 0:
                    logger.info(f"Processing: {content}")
                    result, output_name = pipeline(response=content, sample_name=idx)
                    print(result[output_name])
                    logger.info("Done.")
    else:
        if input_content is not None and len(input_content.strip()) > 0:
            content = input_content.strip()
            logger.info(f"Processing: {content}")
            results = pipeline(content)
            print(results)
            logger.info("Done.")
        else:
            raise ValueError("Invalid input")


if __name__ == '__main__':
    main()
