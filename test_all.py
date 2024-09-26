import subprocess

values = [
          # (34, 0.6, 5, 0.16, 0.6),
          # (34, 0.6, 5, 0.16, 0.7),
          # (34, 0.6, 5, 0.16, 0.8),
          # (34, 0.6, 5, 0.16, 0.9),

          (34, 0.8, 5, 0.08, 0.6),
          (34, 0.8, 5, 0.08, 0.7),
          (34, 0.8, 5, 0.08, 0.8),
          (34, 0.8, 5, 0.08, 0.9),

          (3, 0.6, 5, 0.16, 0.6),
          (3, 0.6, 5, 0.16, 0.7),
          (3, 0.6, 5, 0.16, 0.8),
          (3, 0.6, 5, 0.16, 0.9),
          ]
'''values = [(34, 0.1, 5, 0.36, 0.5),
          (34, 0.2, 5, 0.32, 0.5)]'''

for (attack_type, step_start, step_times, step_weight, motion_val,) in values:
    output_file = f"hm-{attack_type}-st{step_start}-m{motion_val}.txt"

    with open(output_file, 'w') as f:
        subprocess.run(['python', '-u', 'main.py', 'hmdb51', '--evaluate',
                        '--attack_type', str(attack_type), '--step_start', str(step_start),
                        '--step_times', str(step_times), '--step_weight', str(step_weight),
                        '--motion_val', str(motion_val)],
                       stdout=f, stderr=subprocess.STDOUT)

    print('finished:', 'python', '-u', 'main.py', '--evaluate',
          '--attack_type', str(attack_type), '--step_start', str(step_start),
          '--step_times', str(step_times), '--step_weight', str(step_weight),
          '--motion_val', str(motion_val))
