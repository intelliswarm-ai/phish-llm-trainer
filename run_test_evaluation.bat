@echo off
echo Evaluating model on all test datasets...
echo.

python evaluate_test_datasets_wrapper.py

echo.
echo Evaluation complete!
echo Results saved in: test_evaluation_results
echo.
pause