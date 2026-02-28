Create 5 experiments in experiments/xtreme_isolated/

1. PA, NLI, QA, NER, POS - all 5 separate (200 iterations each)
2. Only English language
3. Do a lambda sweep on all - 0(standard SFT), 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1
4. Find the best lambda for each task
5. Run each experiment for 200 iterations only, eval can be done together for each task to reduce model loading overhead. After each eval, update results on xtreme_isolated_results.md. Complete a task before moving to next.
6. One GPU per experiment (4XA100 available) (hence 2 rounds of training for a single task then eval on all 4GPUs at once with tp=4)
7. run parallel experiments in different tmux terminals in ssh cn14-dgx -p 4422 (compute node with shared filesystem)
8. create tmux session named exp1, exp2, exp3, exp4.. send command app(alias for apptainer) then after 5 secs send mamba activate /dev/shm/vllm to activate env on all tmux sessions
9. I need to see live logs in these tmux sessions, you also monitor for any issues and orchestrate the experiments
10. I am assuming 5 tasks * 8 lambda = 40 experiments / 4 gpus = 10 * time for 1 experiment
11. At the end prepare a comprehensive report xtreme_isolated_results.md with 5 plots(for each task) with lambda on x axis and performance on y
12. Give the ideal lambdas at end with your analysis.
13. The end.