import os
import shutil
import glob

#import cv2
#from PIL import Image
#import pytesseract

import easyocr
import torch
from transformers import AutoModel
from sentence_transformers.util import community_detection
#from sentence_transformers.util import pytorch_cos_sim
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import KMeans

################################################################################
# PARAMETROS

PASTA_DAS_IMAGENS = ""
# Only consider cluster that have at least "min_community_size" elements
MIN_COMMUNITY_SIZE = 1
# Consider sentence pairs with a cosine-similarity larger than threshold as similar
THRESHOLD = 0.9

assert PASTA_DAS_IMAGENS

################################################################################
# LISTAR IMAGENS

file_list = glob.glob(PASTA_DAS_IMAGENS)

################################################################################
# EXTRAIR TEXTO DAS IMAGENS

reader = easyocr.Reader(['pt'])
text_list = [' '.join(reader.readtext(file, detail=0)) for file in file_list]

# O reader deixa muita memoria para tras
del reader, easyocr
torch.cuda.empty_cache()

################################################################################
# GERAR EMBEDDINGS

task = "text-matching"
# Modelo escolhido com base no leaderboard de Semantic Textual Similarity (STS).
# O suporte ao portugues e uso de memoria tbm foram criterios usados.
# Link: https://huggingface.co/spaces/mteb/leaderboard
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
embeddings_list = model.encode(text_list, task=task)
del model

################################################################################
# AGRUPAR TEXTOS COM BASE NA SIMILARIDADE

# Only consider cluster that have at least "min_community_size" elements
#min_community_size = 1
# Consider sentence pairs with a cosine-similarity larger than threshold as similar
#threshold = 0.9

# file_list, text_list, embeddings_list
cluster_list = community_detection(
  embeddings_list,
  min_community_size=MIN_COMMUNITY_SIZE,
  threshold=THRESHOLD
)

# sim_matrix = pytorch_cos_sim(embeddings_list, embeddings_list)

################################################################################
# COPIAR ARQUIVOS SIMILARES PARA AS MESMAS PASTAS
# Copia tanto o texto extraido quanto a imagem original

for cluster_idx, cluster in enumerate(cluster_list):
  cluster_name = 'cluster' + str(cluster_idx).zfill(3)
  print('\n' + cluster_name + ':')
  os.mkdir(cluster_name)
  for file_idx in cluster:
    file = file_list[file_idx]
    text = text_list[file_idx]
    #print(' ', file_list[file_idx])
    # Copiar arquivo da imagem original para a pasta do cluster
    img_path = os.path.join(cluster_name, file.replace('/', '_'))
    shutil.copyfile(file, img_path)
    # Salvar texto extraido junto da imagem
    text_path = img_path + str('.txt')
    with open(text_path, 'w+', encoding='utf-8') as f:
      f.write(text)
    print(' ', img_path, text_path)

