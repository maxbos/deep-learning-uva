from multiprocessing.dummy import Pool
from subprocess import run

def get_lr(model_type):
  return 0.001 if model_type == 'RNN' else 0.01

if __name__ == "__main__":
    model_types = ['RNN', 'LSTM']
    cmds = [
      f'/anaconda3/bin/python train.py --device=cpu --input_length={i} --model_type={t} --learning_rate={get_lr(t)}'
      for t in model_types
        for i in range(5, 40)
    ]

    def run_cmd(cmd):
        run(cmd, shell=True)

    p = Pool(4)
    p.map(run_cmd, cmds)
    p.close()
