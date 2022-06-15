import torch.utils.data

import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import gensim
import scipy.io as scio

import os


class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.fc = nn.Linear(2048, 2048)    # 重新定义最后⼀层
        nn.init.eye_(self.model.fc.weight)  # 将⼆维tensor初始化为单位矩阵
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.softmax = nn.Softmax()
        if torch.cuda.is_available():
            self.model.cuda()
            self.softmax.cuda()

    def forward(self, input):
        input_tensor = self.preprocess(input)
        input_tensor = input_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        # 299 x 299 x 3
        with torch.no_grad():
            input_tensor = self.model(input_tensor)
        output = self.softmax(input_tensor[0])
        return output

class CUBDataPreProcessor():
    def __init__(self):
        path_file = "./datasets/birds/CUB_200_2011/images.txt"
        label_file = "./datasets/birds/CUB_200_2011/image_class_labels.txt"
        self.path_list = []
        label_list = []

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label_list.append(line.strip().split(' ')[1])
        self.labels = np.array(label_list).astype(int).reshape((-1, 1))
        print(self.labels.shape)
        with open(path_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.path_list.append(line.strip().split(' ')[1])

    def preProcessText(self):
        docs_list = []
        docs_feat = []
        for doc_path in self.path_list:
            doc_full_path = os.path.join("./datasets/birds/text_c10/", doc_path.replace(".jpg", ".txt"))
            with open(doc_full_path, "r") as f:
                doc = f.read()
                doc_tokens = gensim.utils.simple_preprocess(doc)
                docs_list.append(doc)
        corpus = []
        for i, doc in enumerate(docs_list, 0):
            doc = doc.split()
            taggedDocument = gensim.models.doc2vec.TaggedDocument
            document = taggedDocument(doc, tags=[i])
            corpus.append(document)
        
        d2v_model = gensim.models.doc2vec.Doc2Vec(corpus, vector_size = 300, min_count = 2, epochs = 10)
        d2v_model.train(corpus, total_examples = d2v_model.corpus_count, epochs = d2v_model.epochs)

        for i in range(d2v_model.corpus_count):
            doc_feat = d2v_model.dv[i]
            docs_feat.append(doc_feat)
        docs_feat = np.array(docs_feat)

        print(docs_feat.shape)

        return docs_feat

    def preProcessImage(self):

        inception_v3 = INCEPTION_V3()

        images_feat = []
        for image_path in self.path_list:
            print(image_path)
            image_all_path = os.path.join("./datasets/birds/CUB_200_2011/images/", image_path)
            image = Image.open(image_all_path)
            # 灰度图处理
            if image.layers == 1:
                img = cv2.imread(image_all_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img)

            image_feat = inception_v3(image).cpu().detach().numpy()
            images_feat.append(image_feat)
        images_feat = np.array(images_feat)
        print(images_feat.shape)

        return images_feat

    def preProcessAndSave(self, fileName):
        images_feat = self.preProcessImage()
        docs_feat = self.preProcessText()
        scio.savemat(fileName, {'images_feat': images_feat.astype(float), 'docs_feat': docs_feat.astype(float), 'labels': self.labels})

if __name__ == '__main__':
    preprocessor = CUBDataPreProcessor()
    preprocessor.preProcessAndSave("./datasets/birds/CUB.mat")