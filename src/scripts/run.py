import argparse
import math
import os
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from src.utils.calibration_methods import *
import random
import numpy as np
from datasets import load_dataset



PROCESSER = {
    "sst2": SST2Processor,
    "mnli": MnliProcessor,
    "yahoo_answers_topics": YahooProcessor,
    "amazon_food": AmazonFoodProcessor,
    "civil_comments": CivilCommentsProcessor,
    "dynasent": AmazonFoodProcessor
}

OOD_NAME = {
    "mnli": ["snli", "hans", "anli"],
    "amazon_food": ["sst5", "semeval"],
    "civil_comments": ["hate_speech", "implicit_hate"],
    "sst2": ["wikitext", "random_words"],
    "dynasent": ["dsc", "amazon_food"],
    "yahoo_answers_topics": ["bookcorpus", "random_words"]
}

MODEL_PATH = {
    "t5": "t5-base",
    "t5-base": "t5-base",
    "t5-small": "t5-small",
    "t5-large": "t5-large",
}

DATASET_PATH = {
    "sst2": "./data/TextClassification/SST-2",
    "mnli": "./data/TextClassification/mnli",
    "yahoo_answers_topics": "./data/TextClassification/yahoo_answers_topics",
    "amazon_food": "./data/TextClassification/amazon_food",
    "civil_comments": "./data/TextClassification/civil_comments",
    "dynasent": "./data/TextClassification/dynasent",
}

NUM_CLASSES = {
    "sst2": 2,
    "mnli": 3,
    "yahoo_answers_topics": 10,
    "amazon_food": 3,
    "civil_comments": 2,
    "dynasent": 3
}


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    # rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    # rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    augmented_sentences.append(sentence)
    return augmented_sentences




def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluation(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed):
    allprobs = []
    allpreds = []
    alllabels = []
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            probs = F.softmax(logits, dim=-1)
            if ood_name == "hans" and "calibration" not in method:
                probs = torch.stack([probs[:, 0], probs[:, 1] + probs[:, 2]], dim=1)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            if "calibration" not in method:
                allprobs.extend([prob.max().item() for prob in probs])
            else:
                allprobs.extend([prob[1].item() for prob in probs])
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        os.makedirs(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}", exist_ok=True)
        np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/alllabels.npy", alllabels)
        np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy", allprobs)
        np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allpreds.npy", allpreds)
        acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
        print('acc on {}: {}'.format(ood_name, acc))
        return acc





def LCF(prompt_model, train_dataloader, calibration_data, dataset_name, tokenizer,
                             WrapperClass, vocab=None ):
    # 1. get prediction
    performance_template = prompt_model.template
    performance_verbalizer = prompt_model.verbalizer
    calibartion_template = ManualTemplate(tokenizer=tokenizer).from_file(
        f"scripts/TextClassification/{dataset_name}/calibration_template.txt", choice=0)
    calibartion_verbalizer = ManualVerbalizer(tokenizer, num_classes=2).from_file(
        f"scripts/TextClassification/{dataset_name}/calibration_verbalizer.txt")
    train_dataloader_performance = train_dataloader

    avg_len = 0
    label_words = prompt_model.verbalizer.label_words
    label_count = {0: 0, 1: 0}
    newcalibration_data = deepcopy(calibration_data)
    double = False
    neg_cases_li = []
    if newcalibration_data[0].text_b == '':
        for i in range(len(newcalibration_data)):
            item = newcalibration_data[i]
            avg_len += len(item.text_a.split(' '))
            newcalibration_data[i].text_b = label_words[item.pred][0]
            newcalibration_data[i].label = int(item.flag)
            label_count[int(item.flag)] += 1
            if int(item.flag) == 0:
                neg_cases_li.append(item)
    else:
        double = True
        for i in range(len(newcalibration_data)):
            item = newcalibration_data[i]
            avg_len = 0
            newcalibration_data[i].text_a = "(1) {} (2) {}".format(newcalibration_data[i].text_a, newcalibration_data[i].text_b)
            newcalibration_data[i].text_b = label_words[item.pred][0]
            newcalibration_data[i].label = int(item.flag)
            label_count[int(item.flag)] += 1

            if int(item.flag) == 0:
                neg_cases_li.append(item)



    avg_len /= len(newcalibration_data)
    avg_len = int(avg_len)



    count = 10000
    if neg_sample != '':
        if sample_num == -1:
            negtive_sample_num = label_count[1] - label_count[0]
        else:
            negtive_sample_num = sample_num


        if neg_sample == 'orig':
            for _ in range(negtive_sample_num):
                newcalibration_data.append(random.choice(neg_cases_li))
        elif neg_sample == 'eda':
            for _ in range(negtive_sample_num):
                target_input_example = deepcopy(random.choice(neg_cases_li))
                target_sentence = target_input_example.text_a
                transform_sentence = eda(target_sentence)[0]
                target_input_example.text_a = transform_sentence
                newcalibration_data.append(target_input_example)
        else:
            neg_samples_li = negative_sampling(neg_sample, negtive_sample_num, vocab, avg_len, double)
            for item in neg_samples_li:
                if len(item) == 2:
                    # double
                    example = InputExample(guid=count, text_a="(1) {} (2) {}".format(item[0], item[1]), label=0)
                    example.text_b = random.choice(label_words)[0]
                    count += 1
                    newcalibration_data.append(example)
                else:
                    # single
                    example = InputExample(guid=count, text_a=item, label=0)
                    example.text_b = random.choice(label_words)[0]
                    count += 1
                    newcalibration_data.append(example)

        label_count[0] += negtive_sample_num

    print("label_count: ", label_count)
    print("label_count: ", label_count, file=f, flush=True)


    if clip <= 0:
        shots = min(label_count[0], label_count[1])
        dataset = []
        label_count = {0: 0, 1: clip}
        for data in newcalibration_data:
            if label_count[data.label] < shots:
                dataset.append(data)
                label_count[data.label] += 1
        newcalibration_data = dataset




    print("label_count: ", label_count)
    print("label_count: ", label_count, file=f, flush=True)

    def generate_eda(orig_calibration_data):
        print("Generate EDA data")
        calibration_data_eda = []
        for item in tqdm(orig_calibration_data):
            target_input_example = deepcopy(item)
            target_sentence = target_input_example.text_a
            try:
                transform_sentence = eda(target_sentence)[0]
            except ValueError:
                transform_sentence = target_sentence
            target_input_example.text_a = transform_sentence
            calibration_data_eda.append(target_input_example)
        return calibration_data_eda



    random.shuffle(newcalibration_data)
    calibration_data_eda = generate_eda(newcalibration_data)
    assert len(calibration_data_eda) == len(newcalibration_data)
    train_dataloader_calibration = PromptDataLoader(dataset=newcalibration_data, template=calibartion_template,
                                                            tokenizer=tokenizer,
                                                            tokenizer_wrapper_class=WrapperClass,
                                                            max_seq_length=256, decoder_max_length=3,
                                                            batch_size=16, shuffle=False, teacher_forcing=False,
                                                            predict_eos_token=False,
                                                            truncate_method="tail")
    iter_len = len(train_dataloader_calibration)

    train_dataloader_calibration = iter(train_dataloader_calibration)
    train_dataloader_calibration_eda = iter(PromptDataLoader(dataset=calibration_data_eda, template=calibartion_template,
                                                            tokenizer=tokenizer,
                                                            tokenizer_wrapper_class=WrapperClass,
                                                            max_seq_length=256, decoder_max_length=3,
                                                            batch_size=16, shuffle=False, teacher_forcing=False,
                                                            predict_eos_token=False,
                                                            truncate_method="tail"))



    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()

    a,b = label_count[0] / (label_count[1] + label_count[0]), label_count[1] / (label_count[0] + label_count[1])

    calibration_func = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    #


    prompt_model.train()
    for epoch in range(8):
        tot_loss = 0
        print("Training model for better calibration")
        print("Training model for better calibration", file=f, flush=True)
        prompt_model = PromptForClassification(plm=prompt_model.plm, template=calibartion_template, verbalizer=calibartion_verbalizer, freeze_plm=False).cuda()

        for step in tqdm(range(iter_len)):
            inputs_orig = next(train_dataloader_calibration)
            inputs_eda = next(train_dataloader_calibration_eda)
            inputs_orig = inputs_orig.cuda()
            inputs_eda = inputs_eda.cuda()
            orig_logits = prompt_model(inputs_orig)
            eda_logits = prompt_model(inputs_eda)
            labels = inputs_orig['label']

            loss_orig = calibration_func(orig_logits, labels)

            orig_softmax = F.softmax(orig_logits)
            eda_softmax = F.softmax(eda_logits)

            loss_kl = kl_loss(eda_softmax.log(), orig_softmax)

            loss =  consistent_factor * loss_kl + loss_orig
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Calibration Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
                print("Calibration Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), file=f, flush=True)

        train_dataloader_calibration_eda = iter(
            PromptDataLoader(dataset=calibration_data_eda, template=calibartion_template,
                             tokenizer=tokenizer,
                             tokenizer_wrapper_class=WrapperClass,
                             max_seq_length=256, decoder_max_length=3,
                             batch_size=16, shuffle=False, teacher_forcing=False,
                             predict_eos_token=False,
                             truncate_method="tail"))

        train_dataloader_calibration = iter(PromptDataLoader(dataset=newcalibration_data, template=calibartion_template,
                                                        tokenizer=tokenizer,
                                                        tokenizer_wrapper_class=WrapperClass,
                                                        max_seq_length=256, decoder_max_length=3,
                                                        batch_size=16, shuffle=False, teacher_forcing=False,
                                                        predict_eos_token=False,
                                                        truncate_method="tail"))








        print("Training model for better performance")
        print("Training model for better performance", file=f, flush=True)
        prompt_model = PromptForClassification(plm=prompt_model.plm, template=performance_template,
                                               verbalizer=performance_verbalizer, freeze_plm=False).cuda()
        for step, inputs in enumerate(train_dataloader_performance):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Performance Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
                print("Performance Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), file=f, flush=True)

    return prompt_model, calibartion_template, calibartion_verbalizer


def get_calibration_data(train, hidden):
    def compute_hidden(prompt_model, dataloader):
        all_hidden_states = []
        allpreds = []
        alllabels = []
        prompt_model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader)):
                inputs = inputs.cuda()
                output = prompt_model.prompt_model(inputs)
                batch_size = output.decoder_hidden_states[-1].shape[0]
                hidden_states = output.decoder_hidden_states[-1].reshape((batch_size, -1))
                all_hidden_states.extend(hidden_states.detach().cpu())
                logits = prompt_model(inputs).detach().cpu()
                allpreds.extend(torch.argmax(logits, dim=-1).cpu())
                labels = inputs['label']
                alllabels.extend(labels.cpu())
            dim = output.decoder_hidden_states[-1].shape[1] * output.decoder_hidden_states[-1].shape[2]

        dataset = list(zip(all_hidden_states, allpreds, alllabels))
        random.shuffle(dataset)

        feature_dataset = []
        for hidden_states, pred, label in dataset:
            feature_dataset.append((hidden_states, int(pred == label)))

        return feature_dataset, dim, all_hidden_states, allpreds, alllabels

    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name


    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)

    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(
        f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(
        f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                           freeze_plm=False)
    if torch.cuda.is_available():
        prompt_model.cuda()
    train_dataloader = PromptDataLoader(dataset=train, template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                        batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="tail")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    prompt_model.train()
    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(tqdm(train_dataloader)):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            # if step % 100 == 1:
            #     print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
    loader = PromptDataLoader(dataset=hidden, template=mytemplate, tokenizer=tokenizer,
                                                                   tokenizer_wrapper_class=WrapperClass,
                                                                   max_seq_length=256, decoder_max_length=3,
                                                                   batch_size=16, shuffle=False, teacher_forcing=False,
                                                                   predict_eos_token=False,
                                                                   truncate_method="tail")

    _, _, _, allpreds, alllabels = compute_hidden(prompt_model, loader)
    flag_li = [(allpreds[i] == alllabels[i], allpreds[i]) for i in range(len(allpreds))]
    calibration_data = []
    for i in range(len(flag_li)):
        item = hidden[i]
        item.flag, item.pred = flag_li[i]
        calibration_data.append(item)
    return calibration_data




def negative_sampling(type='random', sample_num=1000, vocab=None, avg_len=None, double=False):
    negative_data = []
    avglen_double = (11,21)
    if type == 'random':
        if double:
            sentence_len_li1 = list(range(11-3, 11+5))
            sentence_len_li2 = list(range(21-5, 21+5))
            for _ in range(sample_num):
                text_a = ' '.join([vocab[np.random.randint(len(vocab))] for _ in range(sentence_len_li1[np.random.randint(8)])])
                text_b = ' '.join([vocab[np.random.randint(len(vocab))] for _ in range(sentence_len_li2[np.random.randint(10)])])
                negative_data.append((text_a, text_b))

        else:
            sentence_len_li = list(range(avg_len - 5, avg_len + 5))
            for _ in range(sample_num):
                negative_data.append(' '.join([vocab[np.random.randint(len(vocab))] for _ in range(sentence_len_li[np.random.randint(10)])]))

    else:
        assert type == 'plain'
        all_num = len(plain_text)
        # corpus[random_id_li[idx]]['text']
        if double:
            sentence_len_li1 = list(range(11 - 3, 11 + 5))
            sentence_len_li2 = list(range(21 - 5, 21 + 5))
            for _ in range(sample_num):
                text_a = plain_text[np.random.randint(all_num)]['text'][:sentence_len_li1[np.random.randint(8)]]
                text_b = plain_text[np.random.randint(all_num)]['text'][:sentence_len_li2[np.random.randint(10)]]
                negative_data.append((text_a, text_b))
        else:
            sentence_len_li = list(range(avg_len - 5, avg_len + 5))
            for _ in range(sample_num):
                negative_data.append(plain_text[np.random.randint(all_num)]['text'][:sentence_len_li[np.random.randint(10)]])


    return negative_data




def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    seed = args.seed
    # exit()
    ood_dataset_path = os.path.join(dataset_path, "ood")

    dataset = {}
    ## Get calibration training data
    processer = PROCESSER[dataset_name]()
    dataset['train'] = processer.get_examples(dataset_path, "train")
    random.shuffle(dataset['train'])
    # assert split_k == 2
    train_len = len(dataset['train'])

    # train, hidden_set = dataset['train'][: train_len // split_k], dataset['train'][train_len // split_k:]
    # calibration_train = processer.get_examples(dataset_path, 'dev')
    # random.shuffle(calibration_train)
    # calibration_train = calibration_train[:3000]
    tmp = dataset['train']

    calibration_data = []
    split_part = train_len // split_k   # 3010, 10, = 300 ,,  3000, 2, 1500
    for i in range(split_k + 1):
        train, hidden_set = dataset['train'][: i * split_part] + dataset['train'][(i+1) * split_part: ], dataset['train'][i * split_part:(i+1)*split_part]
        if len(train) == 0 or len(hidden_set) == 0:
            continue
        calibration_data += get_calibration_data(train, hidden_set)

    vocab = None



    # train, hidden_set = dataset['train'][: train_len // split_k], dataset['train'][train_len // split_k:]






    # calibration_data += get_calibration_data(hidden_set, train)


    ## Begin to train self-calibrators & task-solvers
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
    performancetemplate = ManualTemplate(tokenizer=tokenizer).from_file(
        f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
    performanceverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(
        f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")




    dataset = {}
    dataset['test'] = processer.get_examples(dataset_path, "test")
    for ood_name in OOD_NAME[dataset_name]:
        dataset[ood_name] = processer.get_examples(ood_dataset_path, ood_name)
    dataloader_dict = {}
    for ood_name in dataset.keys():  # including the test split
        test_dataloader = PromptDataLoader(dataset=dataset[ood_name], template=performancetemplate, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                           decoder_max_length=3,
                                           batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                           truncate_method="head")
        dataloader_dict[ood_name] = test_dataloader




    dataset['train'] = tmp
    prompt_model = PromptForClassification(plm=plm, template=performancetemplate, verbalizer=performanceverbalizer,
                                           freeze_plm=False)


    train_dataloader = PromptDataLoader(dataset=dataset['train'], template=performancetemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                        batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="tail")

    classifier, calibartion_template, calibartion_verbalizer = LCF(prompt_model, train_dataloader, calibration_data, dataset_name, tokenizer, WrapperClass, vocab)

    plm = classifier.plm  ### this line is important, to record the tuned backbone model

    method = method_name
    acc_path = f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}"
    os.makedirs(acc_path, exist_ok=True)
    for ood_name, test_dataloader in dataloader_dict.items():  # original dataloader
        if ood_name == "test":
            ood_name = f"{dataset_name}_iid"
        acc = evaluation(test_dataloader, classifier, dataset_name, model_name, ood_name, method, seed)
        np.save(os.path.join(acc_path, f"{ood_name}-{seed}.npy"), acc)

    for ood_name, test_dataloader in dataloader_dict.items():
        test_dataloader = wrap_verbalized_testloader(classifier, test_dataloader, dataset[ood_name],
                                                     calibartion_template, tokenizer, WrapperClass)
        dataloader_dict[ood_name] = test_dataloader
    print("Wrap the calibrater")
    calibrator = PromptForClassification(plm=plm, template=calibartion_template, verbalizer=calibartion_verbalizer,
                                     freeze_plm=False).cuda()

    for ood_name, test_dataloader in dataloader_dict.items():
        if ood_name == "test":
            ood_name = f"{dataset_name}_iid"
        evaluation(test_dataloader, calibrator, dataset_name, model_name, ood_name, method + "-calibration", seed)

    if save_path != '':
        torch.save(plm.state_dict(), str(args.seed) + save_path)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="amazon_food")
    parser.add_argument('--K', type=int, default=2)
    # parser.add_argument('--record', type=str, default='record.txt')
    parser.add_argument('--method_name', type=str, default='LCF')
    parser.add_argument('--neg_sample', type=str, default='', choices=['','random', 'plain', 'orig', 'eda'])
    parser.add_argument('--alpha', type=str, default='False')
    parser.add_argument('--clip', type=int, default=0)
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--scale', default='base')
    args = parser.parse_args()

    f = open(args.method_name + '.txt', 'w')
    consistent_factor = args.factor
    alpha = eval(args.alpha)
    device = torch.device("cuda")
    # clip = eval(args.clip)
    clip = args.clip
    sample_num = args.sample
    save_path = args.save_path
    if args.scale != 'base':
        args.model_name = args.model_name + '-' + args.scale



    args.model_path = MODEL_PATH[args.model_name]
    args.dataset_path = DATASET_PATH[args.dataset_name]
    args.num_classes = NUM_CLASSES[args.dataset_name]
    seed = args.seed
    split_k = args.K
    method_name = args.method_name
    neg_sample = args.neg_sample
    if neg_sample == 'plain':
        plain_text = load_dataset('wikitext', 'wikitext-103-v1')['train']
        plain_text = [{'text': item['text']} for item in plain_text if len(item['text'].split(' ')) > 20]



    id_acc_list = []
    id_ECE_list = []
    id_prob_distribution = []
    ood_acc_list = []
    ood_ECE_list = []
    ood_prob_distribution = []
    if seed != -1:
        set_seed(seed)
        # args.seed= seed
        main(args)
    else:
        for i in range(args.repeats):
            set_seed(i)
            args.seed = i
            main(args)


    f.close()