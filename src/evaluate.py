from SearchGPTService import SearchGPTService
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, help='The task to evaluate')
parser.add_argument('--evaluate_task_data_path', type=str, help='The task data path to evaluate')

args = parser.parse_args()


def main():


    task = args.task
    if task == 'triviaqa':
        from eval.eval import eval
    elif task == 'CHEF_QA':
        from eval.eval import eval
    elif task == 'GOSSIPCOP_SM':
        from summary import eval
    elif task == 'POLITIFACT_SM':
        from summary import eval
    elif task == 'HOVER_QA':
        from eval.eval_retr import eval
    elif task == 'FERVEROUS_QA':
        from eval.eval_retr import eval
    elif task == 'GOSSIPCOP_QA':
        from eval.eval_retr import eval
    elif task == 'GOSSIPCOP_QA_CL':
        from eval.eval_retr import eval
    elif task == 'POLITIFACT_QA_CL':
        from eval.eval_retr import eval
    elif task == 'LIAR_QA':
        from eval.eval import eval
    elif task == 'LIAR_QA_VA':
        from eval.eval_va import eval
    elif task == 'CHEF_QA_VA':
        from eval.eval_va import eval
    elif task == 'LIAR_QA_RETR':
        from eval.eval_retr import eval
    elif task == 'LIAR_QA_RETR_WEB':
        from eval.eval_retr import eval
    elif task == 'CHEF_QA_RETR':
        from eval.eval_retr import eval
    elif task == 'CHEF_QA_RETR_WEB':
        from eval.eval_retr import eval
    elif task == 'CHEF_QA_RETR_WEB_ALL':
        from eval.eval_retr import eval
    elif task == 'POLITIFACT_QA':
        from eval.eval_retr import eval
    elif task == 'WEIBO_QA':
        from eval.eval import eval
    else:
        raise "Task Name Error!"


    print('Start Evaluating...')
    result = eval(args)
    print(f'Acc: {result}')
    print('Evaluate Done')







if __name__=="__main__":
    main()
