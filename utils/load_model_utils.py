import os
import multiprocessing as mp
import time

def load_model_path(path):
    model_dataset = []
    for root, dirs, files in os.walk(path):
        if files:
            for name in files:
                if name.endswith(".pth") and name != "best_val.pth" and not name.startswith("epoch"):
                    model_dataset.append(os.path.join(root, name))

    return model_dataset


def load_models(path, load_model, cuda=False):
    model_list = []
    model_path_list = load_model_path(path)[:10]
    for model_path in model_path_list:
        model = load_model(model_path)
        model.eval()
        if cuda:
            model.cuda()
        model_list.append(model)

    return model_list


def work_fc(load_model, model_path, queue, cuda=False):
    model = load_model(model_path)
    queue.put(model)

def load_models_mp(path, load_model, cuda=False):

    model_list = []
    model_path_list = load_model_path(path)

    # Queue from  Manager() class
    queue = mp.Manager().Queue(500)
    p = mp.Pool(5)

    # start = time.time()
    for model_path in model_path_list:
        # function must not be inner function
        p.apply(work_fc, args=(load_model, model_path, queue, cuda, ))
        # p.apply_async(work_fc, args=(load_model, model_path, queue, cuda, ))
    p.close()
    p.join()

    for i in range(len(model_path_list)):
        model = queue.get()
        model.eval()
        if cuda:
            model.cuda()
        model_list.append(model)
    # print(time.time() - start)
    return model_list