- Use tmux sessions within ssh cn14-dgx -p 4422 (4xA100) for running any compute heavy scripts. 
- Follow the standard tmux sessions with the following names
1. claude - for running most of the compute scripts (primary)
2. vllm - for running any vllm
3. vscode - for parallelizability (you can run any script here when others are busy)
4. tensor - for running tensorboard
- tmux session are started with apptainer using 'app' alias command, then /dev/shm/vllm mamba env is activated. you can create your own tmux session if needed and do these
- make sure the tmux sessions are cd to project root first
- both compute node and login node (local/here) share same filesystem, so anything which doesnt require compute can be run here quickly
- sleep less untill you get things running