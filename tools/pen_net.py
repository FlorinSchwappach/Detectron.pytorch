###############################################################################
# Script for easy management of training processes and testing of resulting nets
#
# There is an environment for every net: a "meta".
# It is represented by a subfolder in the meta_dir.
# 
# Included are:
# + The weight file (.pkl)
# + Config file (.yaml)
# + Tracking of training runs
# + Tracking of test runs
# (+ Reference to dataset version, would require management of dataset versions)
#
# @FlorinSchwappach 19.11.2018

import argparse
import sh
import time
import json
import coolname
import uuid
import os
import sys
from subprocess import Popen, PIPE, STDOUT

pkl_dir = os.path.expanduser("~/Detectron.pytorch/data/pretrained_model/mask-models/")
yaml_dir = os.path.expanduser("~/Detectron.pytorch/configs/baselines/")
meta_base_dir = os.path.expanduser("~/model_meta/")
tool_dir = os.path.expanduser("~/Detectron.pytorch/tools/")

# R_50: ResNet-50 model
# R_101: ResNet-101 model
# X-152-...-IN5k: ResNeXt model trained on ImageNet-5k
baseline_model_names = {"e2e_50_c4_1" : "e2e_mask_rcnn_R-50-C4_1x",   
                "rcnn_50_c4_1" : "mask_rcnn_R-50-C4_1x",
                "e2e_101_fpn_2" : "e2e_mask_rcnn_X-101-32x8d-FPN_2x",
                "e2e_152_fpn_144" : "e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x"}

###############################################################################

# creating new metas: pen_net.py --new_meta e2e_50_c4_1
# creating new meta and run train+test: pen_net.py --new_meta e2e_50_c4_1 --mode train+test
 
#

def parse_args():
    parser = argparse.ArgumentParser(description="Instrumentation of Mask-RCNN")

    # necessary arguments
    parser.add_argument(
            '--mode', dest='run_mode', required=False, help='')

    parser.add_argument(
            '--new_meta', dest='new_model_descriptor', required=False, help='')

    parser.add_argument(
            '--meta', dest='meta_file', required=False, help='')

    # optional arguments
    parser.add_argument(
            '--list_meta', action='store_true', dest='list_meta', required=False, help='')


    parser.add_argument(
            '--list_baseline_models', action='store_true', dest='list_baselines', required=False, help='')

    parser.add_argument(
            '--plot', dest='plot_type', required=False, help='')

    return parser.parse_args()

###############################################################################

def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    if not os.path.isdir(meta_base_dir):
        os.mkdir(meta_base_dir)

    # list all available metainfo configs
    if args.list_meta:
        print(sh.ls(meta_base_dir))
        return 0

    # list all available baseline weight files
    if args.list_baselines:
        for key, value in baseline_model_names.items():
            print(key + ':\t', value)
        return 0

    meta = None

    if args.new_model_descriptor is not None:
        if args.meta_file is not None:
            print("Error. Parameters new_meta and meta are exclusive.")
            return 1

        if args.new_model_descriptor not in baseline_model_names:
            print("Error. RCNN model unknown.")
            return 1
        else:
            # we want a new meta file
            meta = create_new_net_meta(args)

    if args.meta_file is not None:
        with open(meta_base_dir + args.meta_file + '/' + args.meta_file + '.json') as json_file:
            meta = json.load(json_file)

    if args.run_mode == "train+test":
        print("train and test")
        train_uuid = run_training(meta)
        run_test(meta, train_uuid)
    elif args.run_mode == "train":
        print("train")
        run_training(meta)
    elif args.run_mode == "test":
        print("test")
        run_test(meta)
    pass

###############################################################################

def get_path_to_pkl(model_name):
    return pkl_dir + baseline_model_names[model_name] + '.pkl'

def get_path_to_cfg(model_name):
    return yaml_dir + baseline_model_names[model_name] + '.yaml'

def save_meta(filename, net_meta):
     with open(filename, 'w') as outfile:
        json.dump(net_meta, outfile)


def create_new_net_meta(args):
    # we need a baseline weight and config file
    baseline_weight_path = get_path_to_pkl(args.new_model_descriptor)
    baseline_cfg_path = get_path_to_cfg(args.new_model_descriptor)

    # get the date
    created_on = time.strftime("%c")

    # create a name
    net_name = coolname.generate_slug(3)

    # create a folder
    meta_path = os.path.join(meta_base_dir, net_name)
    os.mkdir(meta_path)

    # copy config file for baseline model to it
    new_cfg_path = os.path.join(meta_path, net_name + '.yaml')
    sh.cp(baseline_cfg_path, new_cfg_path)

    # copy weight file to meta dir
    new_weight_path = os.path.join(meta_path, net_name + '.pkl')
    sh.cp(baseline_weight_path, new_weight_path)

    net_meta = {}
    net_meta['net_name'] = net_name
    net_meta['meta_dir'] = meta_path
    net_meta['baseline_weight_path'] = baseline_weight_path
    net_meta['weight_path'] = new_weight_path
    net_meta['created_on'] = created_on
    net_meta['training_runs'] = []
    net_meta['test_runs'] = []
    net_meta['uuid'] = str(uuid.uuid4())
    net_meta['cfg_path'] = new_cfg_path

    meta_filename = os.path.join(meta_path, net_name + '.json')
    save_meta(meta_filename, net_meta)

    print("Created meta " + net_name)

    return net_meta


def run_shell_command(command):
    my_command = ' '.join([str(x) for x in command])
    print("Running " + my_command)
    output = []
    process = Popen(my_command, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        retcode = process.poll()
        line = process.stdout.readline().rstrip().decode('latin-1')
        yield str(line)
        if retcode is not None:
            print("Return Code " + str(retcode))
            break
    pass


def run_training(meta):
    script_output = []
    errorlog = open(os.path.join(meta['meta_dir'], "pen_net_error.log"), "w")
    #outputlog = open("pen_net.log", "w")
    #sys.stdout = outputlog
    sys.stderr = errorlog

    for line in run_shell_command(
            ["python",
            tool_dir + "train_net_step.py",
            "--dataset", "pens",
            "--cfg", meta['cfg_path'],
            "--load_detectron", meta['weight_path'],
            "--use_tfboard",
            # "--nw", 1,
            "--set",
            "OUTPUT_DIR",
            meta['meta_dir']]):
        print(line)
        script_output.append(line)


    if script_output is not None:
        training_run = {}
        training_run['datetime'] = time.strftime("%c")
        training_run['uuid'] = str(uuid.uuid4())
        training_run['logs'] = [script_output]

        output_path = _extract_training_output_paths(script_output)
        print("Output path: " + output_path)
        split_path = output_path.split('/')
        step_name = split_path[-3]

        files_in_step_dir = os.listdir('/'.join(split_path[:-2]))

        # extract name of tensorboard logfile
        logfile = ""
        for filename in files_in_step_dir:
            if 'tfevents' in filename:
                logfile = filename
                break

        training_run['tf_logfile'] = os.path.join(meta['net_name'], logfile)
        training_run['ckpt_path'] = output_path

        meta['training_runs'].append(training_run)
        meta['latest_training_run'] = training_run['uuid']

        meta_filename = os.path.join(meta_base_dir, meta['net_name'], meta['net_name'] + '.json')
        save_meta(meta_filename, meta)

        return training_run['uuid']
    else:
        print("Training command did not give output.")

    errorlog.close()

    pass


def _extract_training_output_paths(script_output):
    output_string = ''.join(script_output)
    search_string = 'save model: '
    wanted_index = output_string.find(search_string)
    if wanted_index == -1:
        print("Error: couldn't find path to training output in script output.")
        return None
    output_path = output_string[wanted_index + len(search_string):]
    return output_path


def _get_latest_training_checkpoint(meta):
    latest_run = None
    for run in meta['training_runs']:
        if run['uuid'] == meta['latest_training_run']:
            latest_run = run
            break
    return latest_run['ckpt_path']


def run_test(meta, training_uuid=None):
    script_output = []

    latest_checkpoint = _get_latest_training_checkpoint(meta)
    ckpt_path = os.path.join(meta['meta_dir'], latest_checkpoint)

    for line in run_shell_command(
            ["python",
             tool_dir + "test_net.py",
             "--dataset", "pens",
             "--cfg", meta['cfg_path'],
             "--load_ckpt", ckpt_path,
             "--multi-gpu-testing",
             "--output_dir", meta['meta_dir']]):
        print(line)
        script_output.append(line)

        #with open(os.path.join(meta['meta_dir'], 'error.log', 'w')) as errorfile:
        #    errorfile.write(e.stderr)
        #errorfile.close()

    if script_output is not None:
        test_run = {}
        test_run['datetime'] = time.strftime("%c")
        test_run['uuid'] = str(uuid.uuid4())
        test_run['logs'] = []
        #test_run['tf_logfile'] = ""

        if training_uuid is not None:
            test_run['linked_training_run'] = training_uuid

        test_run['logs'].append(script_output)

        meta['test_runs'].append(test_run)
        meta['latest_test_run'] = test_run['uuid']

        meta_filename = os.path.join(meta_base_dir, meta['net_name'], meta['net_name'] + '.json')
        save_meta(meta_filename, meta)

        return test_run['uuid']
    else:
        print("Training command did not give output.")

    pass

def run_pen_inference(meta_name, image_path):
    pass


###############################################################################

if __name__ == "__main__":
    main()
