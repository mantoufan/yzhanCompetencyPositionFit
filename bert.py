# -*- coding: utf-8 -*-
"""极简版胜任力模型特征提取与预测：使用bert-lite模型提升性能与效率"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import re
import warnings
from tqdm import tqdm
import time
import os
import pickle
import hashlib
import gzip
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# 全局配置
BERT_MODEL_NAME = "boltuix/bert-lite"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
CACHE_DIR = "./cache"
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, "bert_embeddings_cache.pkl.gz")

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 文本嵌入缓存
text_embedding_cache = {}

# 加载现有缓存文件
def load_embedding_cache():
    global text_embedding_cache
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            print(f"加载嵌入缓存文件: {EMBEDDING_CACHE_FILE}")
            with gzip.open(EMBEDDING_CACHE_FILE, 'rb') as f:
                text_embedding_cache = pickle.load(f)
            print(f"成功加载缓存，包含 {len(text_embedding_cache)} 个嵌入向量")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            text_embedding_cache = {}
    else:
        print("没有找到缓存文件，将创建新缓存")
        text_embedding_cache = {}

# 保存缓存到本地文件
def save_embedding_cache():
    try:
        print(f"保存嵌入缓存到文件: {EMBEDDING_CACHE_FILE}")
        with gzip.open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(text_embedding_cache, f)
        print(f"缓存保存成功，共 {len(text_embedding_cache)} 个嵌入向量")
    except Exception as e:
        print(f"保存缓存失败: {e}")

# 加载BERT模型及Tokenizer
print(f"加载{BERT_MODEL_NAME}模型...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
model.eval()  # 设置为评估模式

# 加载嵌入缓存
load_embedding_cache()

# 基础函数
def modified_jd_df(jd_path):
    tmp_list = []
    with open(jd_path, encoding='utf8') as f:
        for i, j in enumerate(f):
            if i == 175425: j = j.replace('销售\t|置业顾问\t|营销', '销售|置业顾问|营销')
            tmp_list.append(j.split('\t'))
    return pd.DataFrame(tmp_list[1:], columns=tmp_list[0])

def get_salary(x, is_min=True):
    if not isinstance(x, str): return -1
    length = len(x)
    if is_min:
        return int(x[:6]) if length == 12 else int(x[:5]) if length in [10, 11] else int(x[:4]) if length == 9 else -1
    else:
        return int(x[6:]) if length == 12 else int(x[5:]) if length in [10, 11] else int(x[4:]) if length == 9 else -1

def check_columns_exist(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"警告: 缺少列: {missing}")
        return False
    return True

def offline_eval_map(train_df, label, pred_col):
    if not check_columns_exist(train_df, [label, pred_col]): return 0
    try:
        tmp_train = train_df.copy()
        tmp_train['rank'] = tmp_train.groupby('user_id')[pred_col].rank(ascending=False, method='first')
        tmp_x = tmp_train[tmp_train[label] == 1]
        if tmp_x.empty: return 0
        tmp_x[f'{label}_index'] = tmp_x.groupby('user_id')['rank'].rank(ascending=True, method='first')
        tmp_x['score'] = tmp_x[f'{label}_index'] / tmp_x['rank']
        return tmp_x.groupby('user_id')['score'].mean().mean()
    except Exception as e:
        print(f"评估错误: {e}")
        return 0

# 使用MD5哈希文本
def hash_text(text):
    if not isinstance(text, str): return None
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# BERT嵌入函数
def get_bert_embedding(text, max_length=MAX_SEQ_LENGTH):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.config.hidden_size)
    
    # 使用MD5哈希作为缓存键
    cache_key = hash_text(text)
    if cache_key in text_embedding_cache:
        return text_embedding_cache[cache_key]
    
    # 预处理文本
    inputs = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
    
    # 获取BERT嵌入
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    # 缓存嵌入结果
    text_embedding_cache[cache_key] = embedding
    
    # 每增加200个新缓存项就保存一次
    if len(text_embedding_cache) % 200 == 0:
        save_embedding_cache()
        
    return embedding

def process_batch_embeddings(texts, max_length=MAX_SEQ_LENGTH, batch_size=BATCH_SIZE, desc=None):
    if not texts: return []
    
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress_desc = desc if desc else "处理文本嵌入"
    cache_hits = 0
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=progress_desc):
        batch_texts = texts[i:i+batch_size]
        valid_texts, valid_indices, cache_indices, cached_embeddings = [], [], [], []
        
        # 检查哪些文本已在缓存中
        for j, text in enumerate(batch_texts):
            if not isinstance(text, str) or not text.strip(): continue
            cache_key = hash_text(text)
            if cache_key in text_embedding_cache:
                cache_hits += 1
                cache_indices.append(j)
                cached_embeddings.append(text_embedding_cache[cache_key])
            else:
                valid_texts.append(text)
                valid_indices.append(j)
        
        # 处理未缓存的文本
        batch_embeddings = [np.zeros(model.config.hidden_size) for _ in range(len(batch_texts))]
        
        if valid_texts:
            # 预处理未缓存的文本
            inputs = tokenizer(valid_texts, max_length=max_length, padding='max_length', 
                               truncation=True, return_tensors='pt').to(DEVICE)
            
            # 获取BERT嵌入
            with torch.no_grad():
                outputs = model(**inputs)
                valid_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 添加未缓存的嵌入
            for j, idx in enumerate(valid_indices):
                batch_embeddings[idx] = valid_embeddings[j]
                
                # 更新缓存
                if isinstance(batch_texts[idx], str):
                    cache_key = hash_text(batch_texts[idx])
                    text_embedding_cache[cache_key] = valid_embeddings[j]
        
        # 添加已缓存的嵌入
        for j, idx in enumerate(cache_indices):
            batch_embeddings[idx] = cached_embeddings[j]
        
        results.extend(batch_embeddings)
    
    # 打印缓存命中率统计
    if len(texts) > 0:
        print(f"缓存命中率: {cache_hits / len(texts):.2%}")
    
    # 每处理完一批次保存缓存
    if len(valid_texts) > 0:
        save_embedding_cache()
    
    return results

# 胜任力评估函数
def calculate_similarity(emb1, emb2):
    if isinstance(emb1, np.ndarray) and isinstance(emb2, np.ndarray) and emb1.size > 0 and emb2.size > 0:
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0: return np.dot(emb1, emb2) / (norm1 * norm2)
    return 0

def evaluate_skill_match(experience_emb, job_title_emb, job_type_emb, experience_segments=None):
    if not isinstance(experience_emb, np.ndarray) or not isinstance(job_title_emb, np.ndarray): return 0.4
    
    # 计算经验与职位标题和类型的相似度
    title_sim = calculate_similarity(experience_emb, job_title_emb)
    type_sim = calculate_similarity(experience_emb, job_type_emb)
    
    # 如果有分段经验，计算递减加权平均
    if experience_segments and isinstance(experience_segments, list) and len(experience_segments) > 1:
        decay_factor = 0.8
        total_score, weights = 0, []
        
        for i, segment in enumerate(experience_segments):
            if not isinstance(segment, str) or not segment.strip(): continue
            weight = decay_factor ** i
            weights.append(weight)
            
            segment_emb = get_bert_embedding(segment)
            segment_title_sim = calculate_similarity(segment_emb, job_title_emb)
            segment_type_sim = calculate_similarity(segment_emb, job_type_emb)
            
            total_score += weight * max(segment_title_sim, segment_type_sim)
        
        if weights: return total_score / sum(weights)
    
    # 返回标题和类型相似度的加权平均
    return 0.7 * title_sim + 0.3 * type_sim

# 简化评分函数
def score_education_match(cur_degree_num, required_degree_num):
    if np.isnan(cur_degree_num) or np.isnan(required_degree_num): return 0.5
    if cur_degree_num >= required_degree_num:
        return min(1.0 + min((cur_degree_num - required_degree_num) * 0.1, 0.4), 1.4)
    else:
        return max(0.2, 1.0 - (required_degree_num - cur_degree_num) * 0.3)

def score_work_year_match(user_work_year, min_work_year, max_work_year=None):
    if np.isnan(user_work_year) or np.isnan(min_work_year): return 0.5
    max_work_year = max_work_year if not np.isnan(max_work_year if max_work_year is not None else np.nan) else min_work_year * 2.5
    if user_work_year >= min_work_year:
        if user_work_year <= max_work_year: return 1.0
        over_ratio = (user_work_year - max_work_year) / max_work_year
        return max(0.7, 1.0 - 0.2 * min(1, over_ratio))
    else:
        return max(0.3, user_work_year / min_work_year)

def score_salary_match(min_desire, max_desire, min_offer, max_offer):
    if min_desire <= 0 or min_offer <= 0: return 0.5
    if min_offer >= min_desire:
        score = 0.8
        if max_offer >= max_desire and max_desire > 0: score += 0.2
        return score
    else:
        return min(0.6, (min_offer / min_desire) * 0.7)

def score_location_match(live_city, desire_cities, job_city):
    if not isinstance(live_city, (int, str)) or not isinstance(job_city, (int, str)): return 0.5
    live_city, job_city = str(live_city), str(job_city)
    desire_cities = [str(city) for city in desire_cities] if isinstance(desire_cities, list) else []
    if live_city == job_city: return 1.0
    elif job_city in desire_cities: return 0.8
    else: return 0.5

def score_industry_match(cur_industry, desire_industries, jd_industry=None):
    if not isinstance(cur_industry, str) or not isinstance(desire_industries, str): return 0.5
    cur_industries = cur_industry.split(',') if ',' in cur_industry else [cur_industry]
    desire_industries = desire_industries.split(',') if ',' in desire_industries else [desire_industries]
    if set(cur_industries).intersection(set(desire_industries)): return 0.9
    cur_categories = [ind[:2] for ind in cur_industries if len(ind) >= 2]
    desire_categories = [ind[:2] for ind in desire_industries if len(ind) >= 2]
    if set(cur_categories).intersection(set(desire_categories)): return 0.7
    if isinstance(jd_industry, str):
        jd_industries = jd_industry.split(',') if ',' in jd_industry else [jd_industry]
        jd_categories = [ind[:2] for ind in jd_industries if len(ind) >= 2]
        if set(jd_industries).intersection(set(cur_industries)) or set(jd_industries).intersection(set(desire_industries)): return 0.7
        elif set(jd_categories).intersection(set(cur_categories)) or set(jd_categories).intersection(set(desire_categories)): return 0.6
    return 0.4

# 用户行为评估
def evaluate_motivation_and_behavior(df):
    features = {}
    browse_cnt = df['user_jd_cnt'] if isinstance(df['user_jd_cnt'], (int, float)) and not np.isnan(df['user_jd_cnt']) else 0
    features['browse_intensity'] = min(1.0, 0.3 + 0.1 * browse_cnt)
    
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['jd_nunique'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['jd_nunique']) and df['jd_nunique'] > 0:
        features['job_exploration_diversity'] = df['jd_sub_type_nunique'] / df['jd_nunique']
    else:
        features['job_exploration_diversity'] = 0.4
    
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['user_jd_cnt'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['user_jd_cnt']) and df['user_jd_cnt'] > 0:
        features['browse_focus'] = (df['user_jd_cnt'] - df['jd_sub_type_nunique']) / df['user_jd_cnt']
    else:
        features['browse_focus'] = 0.4
    
    return features

# 胜任力维度特征提取
def extract_competency_dimensions(df, text_embeddings=None):
    features = {}
    
    # 检查是否已有预计算的嵌入
    if text_embeddings and isinstance(text_embeddings, dict):
        experience_emb = text_embeddings.get('experience')
        job_title_emb = text_embeddings.get('jd_title')
        job_type_emb = text_embeddings.get('jd_sub_type')
        job_desc_emb = text_embeddings.get('job_description')
    else:
        # 获取文本嵌入
        experience = df.get('experience', '')
        experience_segments = experience.split('|') if isinstance(experience, str) and '|' in experience else None
        
        experience_emb = get_bert_embedding(experience if isinstance(experience, str) else '')
        job_title_emb = get_bert_embedding(df.get('jd_title', ''))
        job_type_emb = get_bert_embedding(df.get('jd_sub_type', ''))
        job_desc_emb = get_bert_embedding(df.get('job_description\n', ''))
    
    # 知识和技能维度
    features['skill_match'] = evaluate_skill_match(
        experience_emb, job_title_emb, job_type_emb, 
        experience_segments if isinstance(df.get('experience'), str) and '|' in df.get('experience') else None
    )
    
    features['education_match'] = score_education_match(
        df.get('cur_degree_id_num', np.nan), 
        df.get('min_edu_level_num', 3) if not np.isnan(df.get('min_edu_level_num', np.nan)) else 3
    ) if 'cur_degree_id_num' in df and not np.isnan(df.get('cur_degree_id_num', np.nan)) else 0.5
    
    features['work_year_match'] = score_work_year_match(
        df.get('user_work_year', np.nan), 
        df.get('min_work_year', 1) if not np.isnan(df.get('min_work_year', np.nan)) else 1,
        df.get('max_work_year', None) if 'max_work_year' in df and not np.isnan(df.get('max_work_year', np.nan)) else None
    ) if isinstance(df.get('user_work_year'), (int, float)) and not np.isnan(df.get('user_work_year', np.nan)) else 0.5
    
    # 社会角色维度
    features['industry_match'] = score_industry_match(
        df.get('cur_industry_id', ''), 
        df.get('desire_jd_industry_id', ''),
        df.get('jd_industry', None) if 'jd_industry' in df and isinstance(df.get('jd_industry'), str) else None
    ) if isinstance(df.get('cur_industry_id'), str) and isinstance(df.get('desire_jd_industry_id'), str) else 0.5
    
    features['job_type_match'] = 1.0 if isinstance(df.get('desire_jd_type_id'), str) and isinstance(df.get('jd_sub_type'), str) and \
                                        df.get('jd_sub_type', '') in df.get('desire_jd_type_id', '') else 0.5
    
    # 自我概念维度
    features['salary_match'] = score_salary_match(
        df.get('min_desire_salary', 0), 
        df.get('max_desire_salary', 0) if 'max_desire_salary' in df and isinstance(df.get('max_desire_salary'), (int, float)) and not np.isnan(df.get('max_desire_salary', np.nan)) else 0,
        df.get('min_salary', 0),
        df.get('max_salary', 0) if 'max_salary' in df and isinstance(df.get('max_salary'), (int, float)) and not np.isnan(df.get('max_salary', np.nan)) else 0
    ) if isinstance(df.get('min_desire_salary'), (int, float)) and isinstance(df.get('min_salary'), (int, float)) and \
         not np.isnan(df.get('min_desire_salary', np.nan)) and not np.isnan(df.get('min_salary', np.nan)) else 0.5
    
    features['location_match'] = score_location_match(
        df.get('live_city_id', -1), 
        df.get('desire_jd_city_id', []) if 'desire_jd_city_id' in df else [],
        df.get('city', -1)
    )
    
    # 特质维度 - 使用BERT嵌入计算相似度
    features['text_similarity'] = calculate_similarity(job_desc_emb, experience_emb)
    
    # 动机维度
    features.update(evaluate_motivation_and_behavior(df))
    
    # 特征交互
    if 'skill_match' in features and 'text_similarity' in features:
        features['skill_text_interaction'] = (features['skill_match'] * 0.6 + features['text_similarity'] * 0.4)
    
    if 'education_match' in features and 'work_year_match' in features:
        features['edu_exp_interaction'] = (features['education_match'] * 0.4 + features['work_year_match'] * 0.6)
    
    # 综合能力匹配
    if all(k in features for k in ['skill_match', 'education_match', 'work_year_match']):
        features['capability_match'] = 0.5 * features['skill_match'] + 0.2 * features['education_match'] + 0.3 * features['work_year_match']
    
    # 归一化
    features = {k: min(max(v, 0), 1.0) for k, v in features.items()}
    return features

def add_competency_features(df):
    print("提取胜任力维度特征...")
    start_time = time.time()
    
    # 预计算文本嵌入
    print("计算文本嵌入...")
    unique_experiences = df['experience'].dropna().unique().tolist()
    unique_jd_titles = df['jd_title'].dropna().unique().tolist()
    unique_jd_sub_types = df['jd_sub_type'].dropna().unique().tolist()
    unique_job_descs = df['job_description\n'].dropna().unique().tolist()
    
    # 并行处理文本嵌入
    n_jobs = max(1, min(os.cpu_count() - 1, 16))
    
    print("步骤1/4: 处理经验文本嵌入")
    experience_embs = process_batch_embeddings(unique_experiences)
    print("步骤2/4: 处理职位标题嵌入")
    jd_title_embs = process_batch_embeddings(unique_jd_titles)
    print("步骤3/4: 处理职位类型嵌入")
    jd_sub_type_embs = process_batch_embeddings(unique_jd_sub_types)
    print("步骤4/4: 处理职位描述嵌入")
    job_desc_embs = process_batch_embeddings(unique_job_descs)

    print(f"所有文本嵌入计算完成，缓存大小: {len(text_embedding_cache)}")
    
    # 创建嵌入字典
    experience_emb_dict = {text: emb for text, emb in zip(unique_experiences, experience_embs) if isinstance(text, str)}
    jd_title_emb_dict = {text: emb for text, emb in zip(unique_jd_titles, jd_title_embs) if isinstance(text, str)}
    jd_sub_type_emb_dict = {text: emb for text, emb in zip(unique_jd_sub_types, jd_sub_type_embs) if isinstance(text, str)}
    job_desc_emb_dict = {text: emb for text, emb in zip(unique_job_descs, job_desc_embs) if isinstance(text, str)}
    
    # 并行处理胜任力特征
    chunk_size = max(1, len(df) // (n_jobs * 2))
    
    def process_chunk(chunk_df):
        results = []
        for _, row in chunk_df.iterrows():
            # 为当前行准备嵌入
            text_embeddings = {
                'experience': experience_emb_dict.get(row.get('experience', ''), np.zeros(model.config.hidden_size)),
                'jd_title': jd_title_emb_dict.get(row.get('jd_title', ''), np.zeros(model.config.hidden_size)),
                'jd_sub_type': jd_sub_type_emb_dict.get(row.get('jd_sub_type', ''), np.zeros(model.config.hidden_size)),
                'job_description': job_desc_emb_dict.get(row.get('job_description\n', ''), np.zeros(model.config.hidden_size))
            }
            results.append(extract_competency_dimensions(row, text_embeddings))
        return results
    
    df_chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in tqdm(df_chunks))
    
    all_features = []
    for chunk_results in results:
        all_features.extend(chunk_results)
    
    # 获取所有特征名称并添加到数据框
    all_feature_names = set()
    for features in all_features:
        all_feature_names.update(features.keys())
    
    for i, feature_name in enumerate(all_feature_names):
        df[f'comp_{feature_name}'] = [features.get(feature_name, 0.5) if i < len(all_features) else 0.5 
                                     for i, features in enumerate(all_features)]
    
    # 标准化处理
    print("标准化胜任力特征...")
    dim_mapping = {
        'knowledge_skills': ['skill_match', 'education_match', 'work_year_match', 'capability_match'],
        'social_role': ['industry_match', 'job_type_match'],
        'self_concept': ['salary_match', 'location_match'],
        'traits': ['text_similarity', 'skill_text_interaction'],
        'motive': ['browse_intensity', 'job_exploration_diversity', 'browse_focus']
    }
    
    # 标准化各维度
    for dim, features in dim_mapping.items():
        dim_cols = [f'comp_{f}' for f in features if f'comp_{f}' in df.columns]
        if dim_cols:
            df[dim_cols] = df[dim_cols].rank(pct=True).clip(0.05, 0.95)
            df[dim_cols] = MinMaxScaler().fit_transform(df[dim_cols].fillna(0.5))
    
    # 计算维度得分
    weights = {
        'knowledge_skills': {'skill_match': 0.5, 'education_match': 0.2, 'work_year_match': 0.2, 'capability_match': 0.1},
        'social_role': {'industry_match': 0.5, 'job_type_match': 0.5},
        'self_concept': {'salary_match': 0.5, 'location_match': 0.5},
        'traits': {'text_similarity': 0.6, 'skill_text_interaction': 0.4},
        'motive': {'browse_intensity': 0.4, 'job_exploration_diversity': 0.3, 'browse_focus': 0.3}
    }
    
    # 计算各维度得分
    for dim, features_weights in weights.items():
        dim_cols, dim_weights = [], []
        for feature, weight in features_weights.items():
            col = f'comp_{feature}'
            if col in df.columns:
                dim_cols.append(col)
                dim_weights.append(weight)
        
        if dim_cols:
            weighted_sum = sum(df[col] * weight for col, weight in zip(dim_cols, dim_weights))
            total_weight = sum(dim_weights)
            df[f'comp_dim_{dim}'] = weighted_sum / total_weight if total_weight > 0 else 0
    
    # 计算整体得分
    dim_importance = {
        'knowledge_skills': 0.25, 'social_role': 0.15, 'self_concept': 0.15, 'traits': 0.25, 'motive': 0.20
    }
    
    df['competency_overall_score'] = sum(df[f'comp_dim_{dim}'] * weight 
                                     for dim, weight in dim_importance.items() 
                                     if f'comp_dim_{dim}' in df.columns)
    
    print(f"胜任力特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    return df

# 模型训练和预测
def train_predict(train_, test_, pred, label, cate_cols, is_shuffle=True, use_cate=True):
    print(f'数据形状: 训练集--{train_.shape} 测试集--{test_.shape}')
    n_splits = 5
    
    # 用户分组
    train_user_id = train_['user_id'].unique()
    if is_shuffle:
        user_groups = {uid: i % n_splits for i, uid in enumerate(train_user_id)}
        train_folds = [([i for i, uid in enumerate(train_user_id) if user_groups[uid] != fold],
                        [i for i, uid in enumerate(train_user_id) if user_groups[uid] == fold])
                       for fold in range(n_splits)]
    else:
        kf = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)
        train_folds = list(kf.split(train_user_id))
    
    sub_preds = np.zeros((test_.shape[0], n_splits))
    train_[f'{label}_pred'] = 0
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = pred
    
    print(f'使用 {len(pred)} 个特征...')
    auc_scores = []
    
    # LightGBM参数
    params = {
        'learning_rate': 0.1, 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
        'num_leaves': 63, 'max_depth': 6, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'min_data_in_leaf': 20, 'lambda_l1': 0.5, 'lambda_l2': 0.5, 'seed': 1, 
        'verbose': -1, 'nthread': -1
    }
    
    for n_fold, (train_idx, valid_idx) in enumerate(train_folds, start=1):
        print(f'第 {n_fold} 折训练开始...')
        
        # 划分训练集和验证集
        train_mask = train_['user_id'].isin(train_user_id[train_idx])
        valid_mask = train_['user_id'].isin(train_user_id[valid_idx])
        
        train_x, train_y = train_.loc[train_mask, pred], train_.loc[train_mask, label]
        valid_x, valid_y = train_.loc[valid_mask, pred], train_.loc[valid_mask, label]
        
        print(f'用于训练的用户:{len(train_idx)} 用于验证的用户:{len(valid_idx)}')
        
        valid_cate_cols = [col for col in cate_cols if col in train_x.columns] if use_cate else []
        dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=valid_cate_cols if use_cate else None)
        dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=valid_cate_cols if use_cate else None)

        # 训练模型
        clf = lgb.train(
            params=params, train_set=dtrain, num_boost_round=1,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(stopping_rounds=1), lgb.log_evaluation(period=100)]
        )
        
        sub_preds[:, n_fold - 1] = clf.predict(test_[pred])
        auc_scores.append(clf.best_score['valid_0']['auc'])
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance()
        train_.loc[valid_mask, f'{label}_pred'] = clf.predict(valid_x)

    # 计算平均特征重要性
    fold_importance_df['avg_imp'] = fold_importance_df[[f'fold_{f}_imp' for f in range(1, n_splits + 1)]].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    fold_importance_df[['Feature', 'avg_imp']].to_csv(f'feat_imp_{label}.csv', index=False, encoding='utf8')
    
    # 使用加权平均的预测值
    auc_weights = np.array(auc_scores) / sum(auc_scores)
    test_[label] = np.sum(sub_preds * auc_weights.reshape(1, -1), axis=1)
    
    print('平均AUC得分:', np.mean(auc_scores))
    return test_[['user_id', 'jd_no', label]], train_[['user_id', 'jd_no', f'{label}_pred', label]], fold_importance_df

def calculate_weights(sat_feat_imp, dev_feat_imp, comp_dims, balance_factor=0.2, sat_ratio=0.7, dim_boost=None):
    # 提取胜任力特征
    comp_features = [f for features in comp_dims.values() for f in features]
    
    # 计算基础权重
    sat_weight = (1 - balance_factor) * sat_ratio
    dev_weight = (1 - balance_factor) * (1 - sat_ratio)
    
    # 计算维度权重
    dim_boost = dim_boost or {dim: 1.0 for dim in comp_dims}
    final_weights = {dim: balance_factor / len(comp_dims) * dim_boost.get(dim, 1.0) for dim in comp_dims}
    
    # 输出权重分配
    print("\n最终模型权重分配:")
    print(f"满意度预测基础权重: {sat_weight:.4f}")
    print(f"投递预测基础权重: {dev_weight:.4f}")
    for dim, weight in final_weights.items():
        print(f"  - {dim}: {weight:.4f} (提升系数: {dim_boost.get(dim, 1.0):.2f})")
    
    return sat_weight, dev_weight, final_weights, \
           sat_feat_imp[sat_feat_imp['Feature'].isin(comp_features)].copy(), \
           dev_feat_imp[dev_feat_imp['Feature'].isin(comp_features)].copy()

# 主函数
if __name__ == "__main__":
    start_time = time.time()
    print(f"程序开始运行: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基础映射表
    min_work_year = {103: 1, 305: 3, 510: 5, 1099: 10}
    max_work_year = {103: 3, 305: 5, 510: 10}
    degree_map = {'其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
                  '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}

    # 路径设置
    sub_path, train_path, test_path = './submit/', './data/', './test/'
    os.makedirs(sub_path, exist_ok=True)
    
    print("加载数据中...")
    # 加载用户数据
    train_user = pd.read_csv(f'{train_path}table1_user.csv', sep=',')
    train_user['desire_jd_city_id'] = train_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    train_user['desire_jd_salary_id'] = train_user['desire_jd_salary_id'].astype(str)
    train_user['min_desire_salary'] = train_user['desire_jd_salary_id'].apply(lambda x: get_salary(x, True))
    train_user['max_desire_salary'] = train_user['desire_jd_salary_id'].apply(lambda x: get_salary(x, False))
    train_user['min_cur_salary'] = train_user['cur_salary_id'].apply(lambda x: get_salary(x, True))
    train_user['max_cur_salary'] = train_user['cur_salary_id'].apply(lambda x: get_salary(x, False))
    train_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)
    
    # 加载职位数据
    train_jd = pd.read_csv(f'{train_path}table2_jd.csv', sep='\t')
    train_jd.drop(['company_name', 'max_edu_level', 'is_mangerial', 'resume_language_required'], axis=1, inplace=True)

    # 处理职位数据
    train_jd['min_work_year'] = train_jd['min_years'].map(min_work_year)
    train_jd['max_work_year'] = train_jd['min_years'].map(max_work_year)
    train_jd['start_date'].replace(r'\N', '22000101', inplace=True)
    train_jd['end_date'].replace(r'\N', '22000101', inplace=True)
    
    # 日期转换
    train_jd['start_date'] = pd.to_datetime(train_jd['start_date'].astype(str).apply(
        lambda x: f'{x[:4]}-{x[4:6]}-{x[6:]}'), errors='coerce')
    train_jd['end_date'] = pd.to_datetime(train_jd['end_date'].astype(str).apply(
        lambda x: f'{x[:4]}-{x[4:6]}-{x[6:]}'), errors='coerce')
    train_jd.loc[train_jd['end_date'] == '2200-01-01', ['start_date', 'end_date']] = np.nan
    
    # 处理职位描述
    print("处理职位描述文本...")
    train_jd['job_description\n'].fillna('', inplace=True)
    
    # 加载行为数据
    print("处理用户行为数据...")
    train_action = pd.read_csv(f'{train_path}table3_action.csv', sep=',')
    
    # 计算统计特征（向量化操作）
    user_jd_groups = train_action.groupby(['user_id', 'jd_no'])
    user_groups = train_action.groupby(['user_id'])
    
    train_action['user_jd_cnt'] = user_jd_groups.size().reset_index(name='count')\
        .set_index(['user_id', 'jd_no'])['count'].reindex(pd.MultiIndex.from_arrays([train_action['user_id'], train_action['jd_no']])).values
    
    train_action['jd_cnt'] = user_groups.size().reset_index(name='count')\
        .set_index('user_id')['count'].reindex(train_action['user_id']).values
        
    train_action['jd_nunique'] = user_groups['jd_no'].nunique().reset_index(name='nunique')\
        .set_index('user_id')['nunique'].reindex(train_action['user_id']).values
    
    # 去重处理
    train_action = train_action.drop_duplicates()
    train_action.sort_values(['user_id', 'jd_no', 'delivered', 'satisfied'], inplace=True)
    train_action = train_action.drop_duplicates(subset=['user_id', 'jd_no'], keep='last')
    train_action = train_action[train_action['jd_no'].isin(train_jd['jd_no'].unique())]

    # 合并训练数据
    print("合并训练数据...")
    train = train_action.merge(train_user, on='user_id', how='left')
    train = train.merge(train_jd, on='jd_no', how='left')
    del train['browsed']

    print('训练数据基础特征已生成...')

    # 加载测试数据
    print("加载测试数据...")
    test_user = pd.read_csv(f'{test_path}user_ToBePredicted.csv', sep='\t')
    test_user['desire_jd_city_id'] = test_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    test_user['desire_jd_salary_id'] = test_user['desire_jd_salary_id'].astype(str)
    test_user['min_desire_salary'] = test_user['desire_jd_salary_id'].apply(lambda x: get_salary(x, True))
    test_user['max_desire_salary'] = test_user['desire_jd_salary_id'].apply(lambda x: get_salary(x, False))
    test_user['min_cur_salary'] = test_user['cur_salary_id'].apply(lambda x: get_salary(x, True))
    test_user['max_cur_salary'] = test_user['cur_salary_id'].apply(lambda x: get_salary(x, False))
    test_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)

    test = pd.read_csv(f'{test_path}zhaopin_round1_user_exposure_B_20190819.csv', sep=' ')
    
    # 与训练数据相同的处理
    user_jd_groups = test.groupby(['user_id', 'jd_no'])
    user_groups = test.groupby(['user_id'])
    
    test['user_jd_cnt'] = user_jd_groups.size().reset_index(name='count')\
        .set_index(['user_id', 'jd_no'])['count'].reindex(pd.MultiIndex.from_arrays([test['user_id'], test['jd_no']])).values
    
    test['jd_cnt'] = user_groups.size().reset_index(name='count')\
        .set_index('user_id')['count'].reindex(test['user_id']).values
        
    test['jd_nunique'] = user_groups['jd_no'].nunique().reset_index(name='nunique')\
        .set_index('user_id')['nunique'].reindex(test['user_id']).values
    
    test = test.drop_duplicates()
    test['delivered'] = -1
    test['satisfied'] = -1

    # 合并测试数据
    test = test.merge(test_user, on='user_id', how='left')
    test = test.merge(train_jd, on='jd_no', how='left')
    print('测试数据基础特征已生成...')

    # 合并所有数据进行特征工程
    print("合并所有数据进行特征工程...")
    all_data = pd.concat([train, test], ignore_index=True)

    # 特征工程
    all_data['jd_user_cnt'] = all_data.groupby(['jd_no'])['user_id'].transform('count').values
    all_data['same_user_city'] = all_data.apply(lambda df: str(df['live_city_id']) in df['desire_jd_city_id'], axis=1).astype(int)
    all_data['city'].fillna(-1, inplace=True)
    all_data['city'] = all_data['city'].astype(int)
    all_data['same_com_live'] = (all_data['city'] == all_data['live_city_id']).astype(int)
    
    # 教育程度处理
    all_data['min_edu_level'] = all_data['min_edu_level'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    all_data['cur_degree_id'] = all_data['cur_degree_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    all_data['min_edu_level_num'] = all_data['min_edu_level'].map(degree_map)
    all_data['cur_degree_id_num'] = all_data['cur_degree_id'].map(degree_map)
    all_data['same_edu'] = (all_data['min_edu_level'] == all_data['cur_degree_id']).astype(int)
    all_data['gt_edu'] = (all_data['cur_degree_id_num'] >= all_data['min_edu_level_num']).astype(int)
    
    # 薪资匹配
    all_data['min_desire_salary_num'] = (all_data['min_desire_salary'] <= all_data['min_salary']).astype(int)
    all_data['min_cur_salary_num'] = (all_data['min_cur_salary'] <= all_data['min_salary']).astype(int)
    all_data['max_desire_salary_num'] = (all_data['max_desire_salary'] <= all_data['max_salary']).astype(int)
    all_data['max_cur_salary_num'] = (all_data['max_cur_salary'] <= all_data['max_salary']).astype(int)
    
    # 行业和职位匹配
    all_data['same_desire_industry'] = all_data.apply(lambda df: df['cur_industry_id'] in df['desire_jd_industry_id'] 
                                                         if isinstance(df['cur_industry_id'], str) and isinstance(df['desire_jd_industry_id'], str) 
                                                         else -1, axis=1).astype(int)
    all_data['same_jd_sub'] = all_data.apply(lambda df: df['jd_sub_type'] in df['desire_jd_type_id'] 
                                                 if isinstance(df['jd_sub_type'], str) and isinstance(df['desire_jd_type_id'], str) 
                                                 else -1, axis=1).astype(int)

    # 时间特征
    all_data['start_month'] = all_data['start_date'].dt.month
    all_data['start_day'] = all_data['start_date'].dt.day
    all_data['end_month'] = all_data['start_date'].dt.month
    all_data['end_day'] = all_data['start_date'].dt.day
    all_data['jd_days'] = (all_data['end_date'] - all_data['start_date']).dt.days

    # 工作经验
    all_data['user_work_year'] = 2019 - all_data['start_work_date'].replace('-', np.nan).astype(float)
    all_data['gt_min_year'] = (all_data['user_work_year'] > all_data['min_work_year']).astype(int)
    all_data['gt_max_year'] = (all_data['user_work_year'] > all_data['max_work_year']).astype(int)
    
    # 经验相关特征
    all_data['len_experience'] = all_data['experience'].apply(lambda x: len(x.split('|')) if isinstance(x, str) else np.nan)
    all_data['desire_jd_industry_id_len'] = all_data['desire_jd_industry_id'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    all_data['desire_jd_type_id_len'] = all_data['desire_jd_type_id'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    
    # 处理经验文本
    all_data['experience'] = all_data['experience'].apply(lambda x: ' '.join(x.split('|') if isinstance(x, str) else ['nan']))
    
    # 删除已使用的列
    all_data.drop(['cur_degree_id_num', 'cur_degree_id', 'min_years',
                   'start_work_date', 'start_date', 'end_date', 'key', 'min_edu_level'], axis=1, inplace=True)

    # 统计特征
    all_data['user_jd_city_nunique'] = all_data.groupby('user_id')['city'].transform('nunique').values
    all_data['jd_user_city_nunique'] = all_data.groupby('jd_no')['live_city_id'].transform('nunique').values
    all_data['jd_title_nunique'] = all_data.groupby('user_id')['jd_title'].transform('nunique').values
    all_data['jd_sub_type_nunique'] = all_data.groupby('user_id')['jd_sub_type'].transform('nunique').values
    
    # 类别特征编码
    for j in ['jd_title', 'jd_sub_type']:
        le = LabelEncoder()
        all_data[j].fillna('nan', inplace=True)
        all_data[f'{j}_map_num'] = le.fit_transform(all_data[j])
    
    # 添加胜任力维度特征 - 使用 bert-lite 模型
    print('添加胜任力维度特征...')
    all_data = add_competency_features(all_data)
    
    # 定义胜任力特征列表
    competency_features = [col for col in all_data.columns if col.startswith('comp_')]
    print(f'生成了{len(competency_features)}个胜任力特征')

    # 定义模型使用的特征
    use_feats = [c for c in all_data.columns if c not in ['user_id', 'jd_no', 'delivered', 'satisfied'] +
                 ['desire_jd_city_id', 'desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id', 'cur_jd_type', 'experience',
                 'jd_title', 'jd_sub_type', 'job_description\n']]
    
    # 确保胜任力特征包含在模型特征中
    for feat in competency_features:
        if feat not in use_feats:
            use_feats.append(feat)
    
    # 定义胜任力维度映射
    competency_dimensions = {
        'knowledge_skills': [col for col in competency_features if any(x in col for x in ['skill', 'education', 'work_year', 'capability'])],
        'social_role': [col for col in competency_features if any(x in col for x in ['industry', 'job_type'])],
        'self_concept': [col for col in competency_features if any(x in col for x in ['salary', 'location'])],
        'traits': [col for col in competency_features if any(x in col for x in ['text_similarity', 'skill_text'])],
        'motive': [col for col in competency_features if any(x in col for x in ['browse', 'exploration', 'focus'])]
    }
    
    # 训练模型
    print('训练满意度(satisfied)预测模型...')
    sub_sat, train_pred_sat, sat_feat_imp = train_predict(
        all_data[all_data['satisfied'] != -1], 
        all_data[all_data['satisfied'] == -1],
        use_feats, 'satisfied', ['live_city_id', 'city']
    )

    print('训练投递(delivered)预测模型...')
    sub_dev, train_pred_dev, dev_feat_imp = train_predict(
        all_data[all_data['delivered'] != -1], 
        all_data[all_data['delivered'] == -1],
        use_feats, 'delivered', ['live_city_id', 'city']
    )
    
    # 评估基础模型效果
    base_train_pred_sat = train_pred_sat.merge(
        all_data[all_data['satisfied'] != -1][['user_id', 'jd_no', 'delivered']],
        on=['user_id', 'jd_no'], how='left'
    )
    
    base_train_pred_sat = base_train_pred_sat.merge(
        train_pred_dev[['user_id', 'jd_no', 'delivered_pred']],
        on=['user_id', 'jd_no'], how='left'
    )
    
    if check_columns_exist(base_train_pred_sat, ['delivered', 'delivered_pred', 'satisfied', 'satisfied_pred']):
        base_dev_map = offline_eval_map(base_train_pred_sat, 'delivered', 'delivered_pred')
        base_sat_map = offline_eval_map(base_train_pred_sat, 'satisfied', 'satisfied_pred')
        print('\n基础模型结果:')
        print(f'dev map: {base_dev_map:.4f}, sat map: {base_sat_map:.4f}, final score: {0.7 * base_sat_map + 0.3 * base_dev_map:.4f}')
    else:
        print("警告: 缺少评估必要的列，跳过基础模型评估")
    
    # 设计维度提升系数
    dim_boost = {
        'knowledge_skills': 1.0, 'social_role': 0.8, 'self_concept': 0.9, 
        'traits': 1.5, 'motive': 1.3  # 提升特质和动机维度的权重
    }
    
    # 使用已知的最优权重系数
    best_factor = 0.15
    
    # 计算权重
    print(f"使用胜任力权重系数: {best_factor}")
    best_sat_weight, best_dev_weight, best_dim_weights, _, _ = calculate_weights(
        sat_feat_imp, dev_feat_imp, competency_dimensions, best_factor, 0.7, dim_boost)
    
    # 合并胜任力维度特征到测试集预测结果
    print("计算最终预测结果...")
    sub_sat_comp = sub_sat.merge(
        all_data[all_data['satisfied'] == -1][['user_id', 'jd_no'] + 
                                            [f'comp_dim_{dim}' for dim in competency_dimensions.keys()] +
                                            ['competency_overall_score']],
        on=['user_id', 'jd_no'], how='left'
    )
    
    # 应用最佳权重计算最终预测得分
    sub_sat_comp['best_merge_prob'] = sub_sat['satisfied'] * best_sat_weight + sub_dev['delivered'] * best_dev_weight
    
    # 添加各胜任力维度的加权分数
    for dim in competency_dimensions.keys():
        dim_col = f'comp_dim_{dim}'
        if dim_col in sub_sat_comp.columns:
            sub_sat_comp['best_merge_prob'] += sub_sat_comp[dim_col].fillna(0.5) * best_dim_weights[dim]
    
    # 生成最终结果
    print("生成最终提交结果...")
    result_list = []
    
    # 使用向量化操作替代循环
    user_groups = sub_sat_comp.groupby('user_id')
    valid_jd_set = set(train_jd['jd_no'])
    
    for user_id, group in user_groups:
        # 将结果分为两部分：在训练集中的JD和不在训练集中的JD
        in_training = group[group['jd_no'].isin(valid_jd_set)].sort_values('best_merge_prob', ascending=False)
        not_in_training = group[~group['jd_no'].isin(valid_jd_set)]
        result_list.append(pd.concat([in_training, not_in_training])[['user_id', 'jd_no', 'best_merge_prob']])
    
    # 合并所有结果
    sub_df = pd.concat(result_list, ignore_index=True)
    
    # 保存最终结果
    sub_df[['user_id', 'jd_no']].to_csv('sub_optimized.csv', index=False)
    
    # 保存特征重要性分析结果
    sat_feat_imp[['Feature', 'avg_imp']].head(50).to_csv('feat_imp_satisfied_top50.csv', index=False, encoding='utf8')
    dev_feat_imp[['Feature', 'avg_imp']].head(50).to_csv('feat_imp_delivered_top50.csv', index=False, encoding='utf8')
    
    # 保存最终缓存
    save_embedding_cache()
    
    # 计算运行时间
    end_time = time.time()
    print(f"\n所有处理完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    
    # 释放BERT模型和缓存的内存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("清理缓存完成，程序结束！")