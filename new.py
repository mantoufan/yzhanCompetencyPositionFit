# -*- coding：utf-8 -*-
# 改进的胜任力模型特征提取与预测 - 性能优化版

import pandas as pd
import numpy as np
import jieba as jb
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import re
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import math
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import Counter

warnings.filterwarnings('ignore')

# 检查列是否存在的辅助函数
def check_columns_exist(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"警告: 缺少列: {missing}")
        return False
    return True

# 基础函数
def modified_jd_df(jd_path):
    tmp_list = []
    with open(jd_path, encoding='utf8') as tmp_file:
        for i, j in enumerate(tmp_file):
            if i == 175425:
                j = j.replace('销售\t|置业顾问\t|营销', '销售|置业顾问|营销')
            tmp = j.split('\t')
            tmp_list.append(tmp)
    return pd.DataFrame(tmp_list[1:], columns=tmp_list[0])

def get_min_salary(x):
    if not isinstance(x, str):
        return -1
    length = len(x)
    if length == 12:
        return int(x[:6])
    elif length == 10:
        return int(x[:5])
    elif length == 11:
        return int(x[:5])
    elif length == 9:
        return int(x[:4])
    else:
        return -1

def get_max_salary(x):
    if not isinstance(x, str):
        return -1
    length = len(x)
    if length == 12:
        return int(x[6:])
    elif length == 10:
        return int(x[5:])
    elif length == 11:
        return int(x[5:])
    elif length == 9:
        return int(x[4:])
    else:
        return -1

def is_same_user_city(df):
    live_city_id = str(df['live_city_id'])
    desire_jd_city = df['desire_jd_city_id']
    return live_city_id in desire_jd_city

# 优化版本，缓存分词结果
word_cache = {}
def jieba_cut_cached(text):
    if not isinstance(text, str) or not text:
        return set()
    if text not in word_cache:
        word_cache[text] = set(jb.cut_for_search(text))
    return word_cache[text]

def jieba_cnt(df):
    experience = df['experience']
    jd_title = df['jd_title']
    jd_sub_type = df['jd_sub_type']
    if not isinstance(experience, str) or not isinstance(jd_sub_type, str):
        return 0
    
    tmp_set = jieba_cut_cached(jd_title) | jieba_cut_cached(jd_sub_type)
    experience_words = jieba_cut_cached(experience)
    
    return len(tmp_set.intersection(experience_words))

def cur_industry_in_desire(df):
    cur_industry_id = df['cur_industry_id']
    desire_jd_industry_id = df['desire_jd_industry_id']
    if isinstance(cur_industry_id, str) and isinstance(desire_jd_industry_id, str):
        return cur_industry_id in desire_jd_industry_id
    else:
        return -1

def desire_in_jd(df):
    desire_jd_type_id = df['desire_jd_type_id']
    jd_sub_type = df['jd_sub_type']
    if isinstance(jd_sub_type, str) and isinstance(desire_jd_type_id, str):
        return jd_sub_type in desire_jd_type_id
    else:
        return -1

def get_tfidf(df, names, merge_id):
    # 更高效的TF-IDF转换
    tfidf_enc_tmp = TfidfVectorizer(ngram_range=(1, 2))
    df[names].fillna('', inplace=True)  # 填充NA值避免错误
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(df[names])
    
    # 使用较低的维度减少计算复杂度
    n_components = min(10, tfidf_vec_tmp.shape[1], tfidf_vec_tmp.shape[0])
    svd_tag_tmp = TruncatedSVD(n_components=n_components, n_iter=20, random_state=2019)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = [f'{names}_svd_{i}' for i in range(n_components)]
    return pd.concat([df[[merge_id]], tag_svd_tmp], axis=1)

def get_str(x):
    if not isinstance(x, str):
        return ''
    return ' '.join([i for i in jb.cut(x) if i not in stop_words])

def offline_eval_map(train_df, label, pred_col):
    """
    高效版MAP评估函数
    """
    # 检查必要列
    if not all(col in train_df.columns for col in [label, pred_col]):
        print(f"错误: 评估需要的列 {label} 或 {pred_col} 不存在")
        return 0
    
    try:
        # 直接使用向量化操作代替apply
        tmp_train = train_df.copy()
        tmp_train['rank'] = tmp_train.groupby('user_id')[pred_col].rank(ascending=False, method='first')
        tmp_x = tmp_train[tmp_train[label] == 1]
        
        if tmp_x.empty:
            print(f"警告: 没有找到标签 {label}=1 的样本")
            return 0
            
        tmp_x[f'{label}_index'] = tmp_x.groupby('user_id')['rank'].rank(ascending=True, method='first')
        tmp_x['score'] = tmp_x[f'{label}_index'] / tmp_x['rank']
        return tmp_x.groupby('user_id')['score'].mean().mean()
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return 0

# 优化：改进的文本相似度计算函数 - 使用缓存提高性能
similarity_cache = {}
def calculate_advanced_text_similarity(text1, text2, stop_words=None):
    """
    使用缓存改进的相似度计算
    """
    # 用于缓存的键
    cache_key = hash((text1, text2))
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]
        
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0
    
    if len(text1) < 2 or len(text2) < 2:
        return 0
        
    if stop_words is None:
        stop_words = []
    
    # 使用缓存的分词结果
    text1_words = [w for w in jieba_cut_cached(text1) if w not in stop_words and len(w.strip()) > 1]
    text2_words = [w for w in jieba_cut_cached(text2) if w not in stop_words and len(w.strip()) > 1]
    
    if not text1_words or not text2_words:
        return 0
    
    # 计算Jaccard相似度
    set1, set2 = set(text1_words), set(text2_words)
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if set1 and set2 else 0
    
    # 简化计算，仅使用关键指标
    counter1 = Counter(text1_words)
    counter2 = Counter(text2_words)
    
    # 计算词频权重
    freq_weight = sum([min(counter1.get(w, 0), counter2.get(w, 0)) for w in set(counter1.keys()) | set(counter2.keys())]) / max(len(text1_words), len(text2_words))
    
    # TF-IDF相似度计算仅在必要时进行
    tfidf_similarity = 0
    if jaccard > 0.1:  # 只有当有基本相似度时才计算TF-IDF
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 1))  # 简化为单词
            tfidf_matrix = vectorizer.fit_transform([' '.join(text1_words), ' '.join(text2_words)])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0
    
    # 综合评分
    result = (0.4 * jaccard + 0.4 * tfidf_similarity + 0.2 * freq_weight)
    similarity_cache[cache_key] = result
    
    return result

# 优化：工作经验质量评估 - 减少重复计算
def evaluate_experience_quality(experience, jd_title, jd_sub_type, stop_words=None):
    """
    评估工作经验的质量和相关性 - 优化版
    """
    if not isinstance(experience, str) or not isinstance(jd_title, str):
        return 0
    
    # 解析经验段落
    exp_segments = experience.split('|') if '|' in experience else [experience]
    
    # 职位名称和类别 - 使用缓存的分词结果
    job_keywords = jieba_cut_cached(jd_title) | jieba_cut_cached(jd_sub_type)
    
    # 提取职位关键词 - 过滤数字和短词
    key_words = [word for word in job_keywords if len(word) >= 2 and not any(c.isdigit() for c in word)]
    
    # 如果没有有效关键词，使用所有词
    if not key_words:
        key_words = list(job_keywords)
    
    # 计算每段经验的相关性并加权
    total_score = 0
    weights = []
    exp_scores = []
    
    # 计算衰减权重系数 - 考虑经验的时间衰减
    decay_factor = 0.8
    
    # 越近期的经验权重越高
    for i, segment in enumerate(exp_segments):
        # 指数衰减权重
        weight = decay_factor ** i  
        weights.append(weight)
        
        # 计算关键词匹配 - 使用缓存的分词结果
        segment_words = jieba_cut_cached(segment)
        if not segment_words:
            exp_scores.append(0)
            continue
            
        # 计算关键词匹配度
        matches = len(job_keywords.intersection(segment_words))
        
        # 计算关键词匹配率
        matching_ratio = matches / math.sqrt(len(segment_words))
        
        # 检查是否包含重要关键词
        if any(key in segment for key in key_words):
            matching_ratio *= 1.2  # 关键词加权
        
        exp_scores.append(matching_ratio)
    
    # 如果没有有效权重，返回0
    if not weights:
        return 0
    
    # 使用加权平均计算总分
    for i, score in enumerate(exp_scores):
        total_score += score * weights[i]
    
    return total_score / sum(weights)

# 优化：教育背景评分函数 - 简化计算
def score_education_match(cur_degree_num, required_degree_num, jd_title=None):
    """
    计算教育背景匹配度的评分 - 简化版
    """
    if np.isnan(cur_degree_num) or np.isnan(required_degree_num):
        return 0.5  # 返回中性值
        
    # 达到或超过要求
    if cur_degree_num >= required_degree_num:
        degree_diff = cur_degree_num - required_degree_num
        # 简化计算，限制上限
        extra_score = min(degree_diff * 0.1, 0.4)
        return min(1.0 + extra_score, 1.4)
    else:
        # 未达到要求，根据差距给予部分分数
        gap = required_degree_num - cur_degree_num
        return max(0.2, 1.0 - gap * 0.3)  # 简化为线性函数

# 优化：工作年限匹配评分函数 - 简化计算
def score_work_year_match(user_work_year, min_work_year, max_work_year=None):
    """
    计算工作年限匹配度的评分 - 简化版
    """
    if np.isnan(user_work_year) or np.isnan(min_work_year):
        return 0.5  # 中性值
    
    # 如果没有最大年限要求，设定上限
    if max_work_year is None or np.isnan(max_work_year):
        max_work_year = min_work_year * 2.5
    
    # 简化计算逻辑
    if user_work_year >= min_work_year:
        if user_work_year <= max_work_year:
            # 在理想范围内
            return 1.0
        else:
            # 超过最大年限，轻微惩罚
            over_ratio = (user_work_year - max_work_year) / max_work_year
            return max(0.7, 1.0 - 0.2 * min(1, over_ratio))
    else:
        # 低于最低要求
        return max(0.3, user_work_year / min_work_year)

# 优化：薪资匹配评分函数 - 简化计算
def score_salary_match(min_desire, max_desire, min_offer, max_offer):
    """
    计算薪资匹配度的评分 - 简化版
    """
    if min_desire <= 0 or min_offer <= 0:
        return 0.5  # 数据不足时返回中性值
    
    # 薪资满足比率
    if min_offer >= min_desire:
        # 最低薪资满足期望
        score = 0.8
        # 如果最高薪资也满足期望，额外加分
        if max_offer >= max_desire and max_desire > 0:
            score += 0.2
        return score
    else:
        # 不满足最低期望
        return min(0.6, (min_offer / min_desire) * 0.7)

# 优化：地域匹配函数 - 简化计算
def score_location_match(live_city, desire_cities, job_city):
    """
    计算地域匹配度评分 - 简化版
    """
    if not isinstance(live_city, (int, str)) or not isinstance(job_city, (int, str)):
        return 0.5
    
    live_city = str(live_city)
    job_city = str(job_city)
    
    # 转换desire_cities为列表格式
    if isinstance(desire_cities, list):
        desire_cities = [str(city) for city in desire_cities]
    else:
        desire_cities = []
    
    # 简化计算逻辑
    if live_city == job_city:
        return 1.0  # 当前居住地与工作地相同
    elif job_city in desire_cities:
        return 0.8  # 工作地在期望地点列表中
    else:
        return 0.5  # 不匹配

# 优化：行业匹配函数 - 简化计算
def score_industry_match(cur_industry, desire_industries, jd_industry=None):
    """
    计算行业匹配度评分 - 简化版
    """
    if not isinstance(cur_industry, str) or not isinstance(desire_industries, str):
        return 0.5
    
    # 分解行业ID
    cur_industries = cur_industry.split(',') if ',' in cur_industry else [cur_industry]
    desire_industries = desire_industries.split(',') if ',' in desire_industries else [desire_industries]
    
    # 精确匹配检查
    if set(cur_industries).intersection(set(desire_industries)):
        return 0.9  # 精确匹配
    
    # 大类匹配检查
    cur_categories = [ind[:2] for ind in cur_industries if len(ind) >= 2]
    desire_categories = [ind[:2] for ind in desire_industries if len(ind) >= 2]
    
    if set(cur_categories).intersection(set(desire_categories)):
        return 0.7  # 大类匹配
    
    # 如果有岗位行业信息，进一步评估
    if isinstance(jd_industry, str):
        jd_industries = jd_industry.split(',') if ',' in jd_industry else [jd_industry]
        jd_categories = [ind[:2] for ind in jd_industries if len(ind) >= 2]
        
        if set(jd_industries).intersection(set(cur_industries)) or set(jd_industries).intersection(set(desire_industries)):
            return 0.7
        elif set(jd_categories).intersection(set(cur_categories)) or set(jd_categories).intersection(set(desire_categories)):
            return 0.6
    
    return 0.4  # 无匹配

# 优化：动机和行为特征评估 - 简化计算
def evaluate_motivation_and_behavior(df):
    """
    评估用户的动机和浏览行为特征 - 简化版
    """
    features = {}
    
    # 浏览强度 - 简化计算
    if isinstance(df['user_jd_cnt'], (int, float)) and not np.isnan(df['user_jd_cnt']):
        browse_cnt = df['user_jd_cnt']
        features['browse_intensity'] = min(1.0, 0.3 + 0.1 * browse_cnt)
    else:
        features['browse_intensity'] = 0.3
    
    # 职位探索多样性 - 简化计算
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['jd_nunique'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['jd_nunique']) and df['jd_nunique'] > 0:
        
        # 计算多样性比例
        diversity_ratio = df['jd_sub_type_nunique'] / df['jd_nunique']
        features['job_exploration_diversity'] = diversity_ratio
    else:
        features['job_exploration_diversity'] = 0.4
    
    # 浏览稳定性和专注度 - 简化为一个特征
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['user_jd_cnt'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['user_jd_cnt']) and df['user_jd_cnt'] > 0:
        
        # 计算职位类型的重复浏览率
        features['browse_focus'] = (df['user_jd_cnt'] - df['jd_sub_type_nunique']) / df['user_jd_cnt']
    else:
        features['browse_focus'] = 0.4
    
    return features

# 优化：提取胜任力维度特征的函数 - 减少计算
def extract_competency_dimensions(df, stop_words=None):
    """
    提取胜任力各维度特征 - 优化版
    """
    features = {}
    
    # 1. 知识和技能维度特征
    if isinstance(df['experience'], str) and isinstance(df['jd_title'], str) and isinstance(df['jd_sub_type'], str):
        features['skill_match'] = evaluate_experience_quality(df['experience'], df['jd_title'], df['jd_sub_type'], stop_words)
    else:
        features['skill_match'] = 0.4
    
    # 教育背景评分
    if 'cur_degree_id_num' in df and not np.isnan(df['cur_degree_id_num']):
        min_edu_num = df['min_edu_level_num'] if not np.isnan(df['min_edu_level_num']) else 3
        features['education_match'] = score_education_match(df['cur_degree_id_num'], min_edu_num)
    else:
        features['education_match'] = 0.5
    
    # 工作年限评分
    if isinstance(df['user_work_year'], (int, float)) and not np.isnan(df['user_work_year']):
        min_year = df['min_work_year'] if not np.isnan(df['min_work_year']) else 1
        max_year = df['max_work_year'] if 'max_work_year' in df and not np.isnan(df['max_work_year']) else None
        features['work_year_match'] = score_work_year_match(df['user_work_year'], min_year, max_year)
    else:
        features['work_year_match'] = 0.5
    
    # 2. 社会角色维度特征
    # 行业匹配
    if isinstance(df['cur_industry_id'], str) and isinstance(df['desire_jd_industry_id'], str):
        jd_industry = df['jd_industry'] if 'jd_industry' in df and isinstance(df['jd_industry'], str) else None
        features['industry_match'] = score_industry_match(df['cur_industry_id'], df['desire_jd_industry_id'], jd_industry)
    else:
        features['industry_match'] = 0.5
    
    # 职位类型匹配
    if isinstance(df['desire_jd_type_id'], str) and isinstance(df['jd_sub_type'], str):
        direct_match = df['jd_sub_type'] in df['desire_jd_type_id']
        features['job_type_match'] = 1.0 if direct_match else 0.5
    else:
        features['job_type_match'] = 0.5
    
    # 3. 自我概念维度特征
    # 薪资期望匹配
    if isinstance(df['min_desire_salary'], (int, float)) and isinstance(df['min_salary'], (int, float)) and \
       not np.isnan(df['min_desire_salary']) and not np.isnan(df['min_salary']):
        max_desire = df['max_desire_salary'] if 'max_desire_salary' in df and \
                    isinstance(df['max_desire_salary'], (int, float)) and not np.isnan(df['max_desire_salary']) else 0
        max_offer = df['max_salary'] if 'max_salary' in df and \
                   isinstance(df['max_salary'], (int, float)) and not np.isnan(df['max_salary']) else 0
        features['salary_match'] = score_salary_match(df['min_desire_salary'], max_desire, df['min_salary'], max_offer)
    else:
        features['salary_match'] = 0.5
    
    # 地点偏好匹配
    features['location_match'] = score_location_match(
        df['live_city_id'], 
        df['desire_jd_city_id'] if 'desire_jd_city_id' in df else [], 
        df['city']
    )
    
    # 4. 特质维度特征
    # 职位描述与经验的文本相似度
    job_desc = df['job_description\n'] if isinstance(df['job_description\n'], str) else ""
    experience = df['experience'] if isinstance(df['experience'], str) else ""
    features['text_similarity'] = calculate_advanced_text_similarity(job_desc, experience, stop_words)
    
    # 5. 动机维度特征
    motivation_features = evaluate_motivation_and_behavior(df)
    features.update(motivation_features)
    
    # 6. 特征交互 - 简化计算
    # 技能与经验文本的综合评分
    if 'skill_match' in features and 'text_similarity' in features:
        features['skill_text_interaction'] = (features['skill_match'] * 0.6 + features['text_similarity'] * 0.4)
    
    # 教育与工作年限的综合评分
    if 'education_match' in features and 'work_year_match' in features:
        features['edu_exp_interaction'] = (features['education_match'] * 0.4 + features['work_year_match'] * 0.6)
    
    # 7. 综合能力匹配
    if 'skill_match' in features and 'education_match' in features and 'work_year_match' in features:
        features['capability_match'] = 0.5 * features['skill_match'] + 0.2 * features['education_match'] + 0.3 * features['work_year_match']
    
    # 归一化特征值到0-1范围
    for key in features:
        if features[key] > 1.0:
            features[key] = 1.0
        elif features[key] < 0:
            features[key] = 0.0
    
    return features

# 优化：为数据集添加胜任力维度特征的函数 - 并行处理加速
def add_competency_dimension_features(df, stop_words=None):
    """为数据集添加胜任力维度特征 - 优化版"""
    print("提取胜任力维度特征...")
    start_time = time.time()
    
    # 使用并行处理加速特征提取
    n_jobs = max(1, min(8, os.cpu_count() - 1))  # 限制并行任务数
    chunk_size = max(1, len(df) // (n_jobs * 2))  # 动态确定分片大小
    
    # 分块处理函数
    def process_chunk(chunk_df):
        return chunk_df.apply(lambda x: extract_competency_dimensions(x, stop_words), axis=1)
    
    # 分割数据
    df_chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # 并行处理
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in tqdm(df_chunks))
    
    # 合并结果
    competency_features = pd.concat(results)
    
    # 获取所有特征名称
    all_feature_names = set()
    for features in competency_features:
        all_feature_names.update(features.keys())
    
    # 将特征添加到数据框
    for feature_name in all_feature_names:
        df[f'comp_{feature_name}'] = competency_features.apply(lambda x: x.get(feature_name, 0.5))
    
    # 标准化处理
    print("标准化胜任力特征...")
    dim_mapping = {
        'knowledge_skills': ['skill_match', 'education_match', 'work_year_match', 'capability_match'],
        'social_role': ['industry_match', 'job_type_match'],
        'self_concept': ['salary_match', 'location_match'],
        'traits': ['text_similarity', 'skill_text_interaction'],
        'motive': ['browse_intensity', 'job_exploration_diversity', 'browse_focus']
    }
    
    # 为每个维度标准化
    for dim, features in dim_mapping.items():
        dim_cols = [f'comp_{f}' for f in features if f'comp_{f}' in df.columns]
        if dim_cols:
            df[dim_cols] = df[dim_cols].rank(pct=True).clip(0.05, 0.95)
            scaler = MinMaxScaler()
            df[dim_cols] = scaler.fit_transform(df[dim_cols].fillna(0.5))
    
    # 计算维度综合得分
    weights = {
        'knowledge_skills': {'skill_match': 0.5, 'education_match': 0.2, 'work_year_match': 0.2, 'capability_match': 0.1},
        'social_role': {'industry_match': 0.5, 'job_type_match': 0.5},
        'self_concept': {'salary_match': 0.5, 'location_match': 0.5},
        'traits': {'text_similarity': 0.6, 'skill_text_interaction': 0.4},
        'motive': {'browse_intensity': 0.4, 'job_exploration_diversity': 0.3, 'browse_focus': 0.3}
    }
    
    # 使用加权平均计算各维度得分
    for dim, features_weights in weights.items():
        dim_cols = []
        dim_weights = []
        
        for feature, weight in features_weights.items():
            col = f'comp_{feature}'
            if col in df.columns:
                dim_cols.append(col)
                dim_weights.append(weight)
        
        if dim_cols:
            # 直接计算加权平均
            weighted_sum = sum(df[col] * weight for col, weight in zip(dim_cols, dim_weights))
            total_weight = sum(dim_weights)
            df[f'comp_dim_{dim}'] = weighted_sum / total_weight if total_weight > 0 else 0
    
    # 计算整体胜任力得分
    dim_importance = {
        'knowledge_skills': 0.25,
        'social_role': 0.15,
        'self_concept': 0.15,
        'traits': 0.25, 
        'motive': 0.20
    }
    
    # 直接使用向量化计算整体得分
    df['competency_overall_score'] = 0
    for dim, weight in dim_importance.items():
        col = f'comp_dim_{dim}'
        if col in df.columns:
            df['competency_overall_score'] += df[col] * weight
    
    print(f"胜任力特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    return df

# 优化：更高效的特征重要性分析
def analyze_feature_importance(imp_df, competency_features, top_n=20):
    """简化版特征重要性分析函数"""
    # 只保留有实际重要性的特征
    imp_df = imp_df[imp_df['avg_imp'] > 0].copy()
    
    competency_imp = imp_df[imp_df['Feature'].isin(competency_features)].copy()
    total_comp_imp = competency_imp['avg_imp'].sum()
    total_imp = imp_df['avg_imp'].sum()
    
    print(f"胜任力特征总重要性: {total_comp_imp:.4f}, 占比: {total_comp_imp/total_imp:.2%}")
    
    # 按维度分组分析
    dim_mapping = {
        '知识技能维度': ['comp_skill_match', 'comp_education_match', 'comp_work_year_match', 
                    'comp_capability_match'],
        '社会角色维度': ['comp_industry_match', 'comp_job_type_match'],
        '自我概念维度': ['comp_salary_match', 'comp_location_match'],
        '特质维度': ['comp_text_similarity', 'comp_skill_text_interaction'],
        '动机维度': ['comp_browse_intensity', 'comp_job_exploration_diversity', 'comp_browse_focus']
    }
    
    # 添加维度综合特征
    for dim, features in dim_mapping.items():
        dim_name = dim.replace('维度', '')
        dim_mapping[dim].append(f'comp_dim_{dim_name}')
    
    print("\n各胜任力维度重要性:")
    dim_importances = {}
    
    for dim, features in dim_mapping.items():
        dim_features = [f for f in features if f in competency_features]
        if dim_features:
            dim_imp = competency_imp[competency_imp['Feature'].isin(dim_features)]['avg_imp'].sum()
            dim_importances[dim] = dim_imp
            print(f"  {dim}: {dim_imp:.4f}, 占比: {dim_imp/total_imp:.2%}")
    
    # 显示所有胜任力特征中最重要的N个
    print(f"\n前{top_n}个最重要的胜任力特征:")
    top_comp_features = competency_imp.sort_values('avg_imp', ascending=False).head(top_n)
    for _, row in top_comp_features.iterrows():
        print(f"  {row['Feature']}: {row['avg_imp']:.4f}, 占比: {row['avg_imp']/total_imp:.2%}")
    
    return competency_imp, dim_importances

# 优化：模型训练和预测函数 - 减少冗余计算
def sub_on_line(train_, test_, pred, label, cate_cols, is_shuffle=True, use_cate=True):
    print(f'数据形状:\n训练集--{train_.shape}\n测试集--{test_.shape}')
    n_splits = 5
    
    # 优化：避免不必要的深拷贝
    train_user_id = train_['user_id'].unique()
    
    # 使用用户ID分组进行折叠
    if is_shuffle:
        # 为每个用户分配一个组
        user_groups = {uid: i % n_splits for i, uid in enumerate(train_user_id)}
        
        # 创建用户分组索引
        train_folds = []
        for fold in range(n_splits):
            train_idx = [i for i, uid in enumerate(train_user_id) if user_groups[uid] != fold]
            valid_idx = [i for i, uid in enumerate(train_user_id) if user_groups[uid] == fold]
            train_folds.append((train_idx, valid_idx))
    else:
        # 使用标准KFold
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)
        train_folds = list(folds.split(train_user_id))
    
    sub_preds = np.zeros((test_.shape[0], n_splits))
    train_[f'{label}_pred'] = 0
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = pred
    
    print(f'使用 {len(pred)} 个特征 ...')
    auc_scores = []
    
    # 优化LightGBM参数
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,  # 减少复杂度
        'max_depth': 6,    # 减少深度
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'seed': 1,
        'verbose': -1,
        'nthread': -1      # 使用所有CPU
    }
    
    for n_fold, (train_idx, valid_idx) in enumerate(train_folds, start=1):
        print(f'第 {n_fold} 折训练开始 ...')
        
        # 优化：使用布尔索引代替isin可能更快
        train_mask = train_['user_id'].isin(train_user_id[train_idx])
        valid_mask = train_['user_id'].isin(train_user_id[valid_idx])
        
        train_x, train_y = train_.loc[train_mask, pred], train_.loc[train_mask, label]
        valid_x, valid_y = train_.loc[valid_mask, pred], train_.loc[valid_mask, label]
        
        print(f'用于训练的用户:{len(train_idx)}\n用于验证的用户:{len(valid_idx)}')
        
        # 处理类别特征 - 只处理实际存在的列
        if use_cate:
            valid_cate_cols = [col for col in cate_cols if col in train_x.columns]
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=valid_cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=valid_cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        # 训练模型 - 保持1轮迭代
        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=1,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=1),
                lgb.log_evaluation(period=1)
            ]
        )
        
        # 预测和保存结果
        sub_preds[:, n_fold - 1] = clf.predict(test_[pred])
        auc_scores.append(clf.best_score['valid_0']['auc'])
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance()
        
        # 保存验证集预测结果
        train_.loc[valid_mask, f'{label}_pred'] = clf.predict(valid_x)

    # 计算平均特征重要性
    fold_importance_df['avg_imp'] = fold_importance_df[[f'fold_{f}_imp' for f in range(1, n_splits + 1)]].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    
    # 保存精简版特征重要性
    fold_importance_df[['Feature', 'avg_imp']].to_csv(f'feat_imp_{label}.csv', index=False, encoding='utf8')
    
    # 使用加权平均的预测值
    auc_weights = np.array(auc_scores) / sum(auc_scores)
    test_[label] = np.sum(sub_preds * auc_weights.reshape(1, -1), axis=1)
    
    print('平均AUC得分:', np.mean(auc_scores))
    return test_[['user_id', 'jd_no', label]], train_[['user_id', 'jd_no', f'{label}_pred', label]], fold_importance_df

# 优化：简化权重计算
def calculate_adaptive_weights(sat_feat_imp, dev_feat_imp, competency_dimensions, 
                             balance_factor=0.2, sat_ratio=0.7, dim_boost=None):
    """
    计算各维度的自适应权重 - 简化版
    """
    # 提取胜任力特征
    competency_features = []
    for features in competency_dimensions.values():
        competency_features.extend(features)
    
    # 获取重要性
    comp_sat_imp = sat_feat_imp[sat_feat_imp['Feature'].isin(competency_features)].copy()
    comp_dev_imp = dev_feat_imp[dev_feat_imp['Feature'].isin(competency_features)].copy()
    
    # 计算维度权重
    dim_weights_sat = {}
    dim_weights_dev = {}
    
    # 维度映射
    dim_name_map = {
        'knowledge_skills': '知识技能维度',
        'social_role': '社会角色维度',
        'self_concept': '自我概念维度',
        'traits': '特质维度',
        'motive': '动机维度'
    }
    
    # 使用默认的维度提升系数
    if dim_boost is None:
        dim_boost = {dim: 1.0 for dim in competency_dimensions}
    
    # 计算基础权重
    sat_weight = (1 - balance_factor) * sat_ratio
    dev_weight = (1 - balance_factor) * (1 - sat_ratio)
    
    # 计算简化版维度权重
    final_dim_weights = {}
    for dim in competency_dimensions:
        # 使用固定的维度权重分配
        base_weight = balance_factor / len(competency_dimensions)
        boost = dim_boost.get(dim, 1.0)
        final_dim_weights[dim] = base_weight * boost
    
    # 输出权重分配情况
    print("\n最终模型权重分配:")
    print(f"满意度预测基础权重: {sat_weight:.4f}")
    print(f"投递预测基础权重: {dev_weight:.4f}")
    
    for dim, weight in final_dim_weights.items():
        print(f"  - {dim}: {weight:.4f} (提升系数: {dim_boost.get(dim, 1.0):.2f})")
    
    return sat_weight, dev_weight, final_dim_weights, comp_sat_imp, comp_dev_imp

# 主函数优化
if __name__ == "__main__":
    import os
    from datetime import datetime
    
    # 记录开始时间
    start_time = time.time()
    print(f"程序开始运行: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基础映射表
    min_work_year = {103: 1, 305: 3, 510: 5, 1099: 10}
    max_work_year = {103: 3, 305: 5, 510: 10}
    degree_map = {'其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
                  '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}

    # 路径设置
    sub_path = './submit/'
    train_data_path = './data/'
    test_data_path = './test/'
    os.makedirs(sub_path, exist_ok=True)
    
    print("加载数据中...")
    # 加载用户数据
    train_user = pd.read_csv(train_data_path + 'table1_user.csv', sep=',')
    train_user['desire_jd_city_id'] = train_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    train_user['desire_jd_salary_id'] = train_user['desire_jd_salary_id'].astype(str)
    train_user['min_desire_salary'] = train_user['desire_jd_salary_id'].apply(get_min_salary)
    train_user['max_desire_salary'] = train_user['desire_jd_salary_id'].apply(get_max_salary)
    train_user['min_cur_salary'] = train_user['cur_salary_id'].apply(get_min_salary)
    train_user['max_cur_salary'] = train_user['cur_salary_id'].apply(get_max_salary)
    train_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)
    
    # 加载职位数据
    train_jd = pd.read_csv(train_data_path + 'table2_jd.csv', sep='\t')
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

    print('加载停用词列表...')
    stop_words = [i.strip() for i in open('中文停用词表.txt', 'r', encoding='utf8').readlines()]
    stop_words.extend(['\n', '\xa0', '\u3000', '\u2002'])
    
    print("处理职位描述文本...")
    # 使用并行处理提高效率
    train_jd['job_description\n'].fillna('', inplace=True)
    tmp_cut = Parallel(n_jobs=-1)(delayed(get_str)(train_jd.loc[ind]['job_description\n'])
                                 for ind in tqdm(train_jd.index))

    # 仅当数据量不太大时才进行TF-IDF处理
    if len(tmp_cut) <= 100000:  # 限制处理量
        print("执行TF-IDF分析...")
        tfidf_enc = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)  # 限制特征数
        tfidf_vec = tfidf_enc.fit_transform(tmp_cut)
        
        n_components = min(10, tfidf_vec.shape[1], tfidf_vec.shape[0])
        svd_tag = TruncatedSVD(n_components=n_components, n_iter=20, random_state=2019)
        tag_svd = svd_tag.fit_transform(tfidf_vec)
        
        tag_svd = pd.DataFrame(tag_svd)
        tag_svd.columns = [f'desc_svd_{i}' for i in range(n_components)]
        train_jd = pd.concat([train_jd, tag_svd], axis=1)
    
    # 加载行为数据
    print("处理用户行为数据...")
    train_action = pd.read_csv(train_data_path + 'table3_action.csv', sep=',')
    
    # 使用向量化操作代替groupby transform
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

    # 合并数据
    print("合并训练数据...")
    train = train_action.merge(train_user, on='user_id', how='left')
    train = train.merge(train_jd, on='jd_no', how='left')
    del train['browsed']  # 删除无用列

    print('训练数据基础特征已生成...')

    # 加载测试数据
    print("加载测试数据...")
    test_user = pd.read_csv(test_data_path + 'user_ToBePredicted.csv', sep='\t')
    test_user['desire_jd_city_id'] = test_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    test_user['desire_jd_salary_id'] = test_user['desire_jd_salary_id'].astype(str)
    test_user['min_desire_salary'] = test_user['desire_jd_salary_id'].apply(get_min_salary)
    test_user['max_desire_salary'] = test_user['desire_jd_salary_id'].apply(get_max_salary)
    test_user['min_cur_salary'] = test_user['cur_salary_id'].apply(get_min_salary)
    test_user['max_cur_salary'] = test_user['cur_salary_id'].apply(get_max_salary)
    test_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)

    test = pd.read_csv(test_data_path + 'zhaopin_round1_user_exposure_B_20190819.csv', sep=' ')
    
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

    # 合并所有数据
    print("合并所有数据进行特征工程...")
    all_data = pd.concat([train, test], ignore_index=True)

    # 特征工程 - 使用向量化操作
    print("生成特征...")
    all_data['jd_user_cnt'] = all_data.groupby(['jd_no'])['user_id'].transform('count').values
    all_data['same_user_city'] = all_data.apply(is_same_user_city, axis=1).astype(int)
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
    all_data['same_desire_industry'] = all_data.apply(cur_industry_in_desire, axis=1).astype(int)
    all_data['same_jd_sub'] = all_data.apply(desire_in_jd, axis=1).astype(int)

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
    all_data['len_experience'] = all_data['experience'].apply(
        lambda x: len(x.split('|')) if isinstance(x, str) else np.nan)
    
    all_data['desire_jd_industry_id_len'] = all_data['desire_jd_industry_id'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    
    all_data['desire_jd_type_id_len'] = all_data['desire_jd_type_id'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    
    # 计算经验与职位的匹配
    print("计算经验匹配度...")
    all_data['eff_exp_cnt'] = all_data.apply(jieba_cnt, axis=1)
    all_data['eff_exp_ratio'] = all_data['eff_exp_cnt'] / all_data['len_experience']
    
    # 删除已使用的列
    all_data.drop(['cur_degree_id_num', 'cur_degree_id', 'desire_jd_city_id', 'min_years',
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

    # 处理经验文本
    all_data['experience'] = all_data['experience'].apply(lambda x: ' '.join(x.split('|') if
                                                                            isinstance(x, str) else ['nan']))
    
    # 获取经验TF-IDF特征
    if len(all_data) <= 100000:  # 限制处理量
        print("提取经验TF-IDF特征...")
        exp_gp = all_data.groupby('jd_no')['experience'].agg(lambda x: ' '.join(x.to_list())).reset_index()
        exp_gp = get_tfidf(exp_gp, 'experience', 'jd_no')
        all_data = all_data.merge(exp_gp, on='jd_no', how='left')
    
    # 添加胜任力维度特征
    print('添加胜任力维度特征...')
    all_data = add_competency_dimension_features(all_data, stop_words)
    
    # 定义胜任力特征列表
    competency_features = [col for col in all_data.columns if col.startswith('comp_')]
    print(f'生成了{len(competency_features)}个胜任力特征')

    # 定义模型使用的特征
    use_feats = [c for c in all_data.columns if c not in ['user_id', 'jd_no', 'delivered', 'satisfied'] +
                 ['desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id', 'cur_jd_type', 'experience',
                 'jd_title', 'jd_sub_type', 'job_description\n']]
    
    # 确保胜任力特征包含在模型特征中
    for feat in competency_features:
        if feat not in use_feats:
            use_feats.append(feat)
    
    # 定义胜任力维度映射
    competency_dimensions = {
        'knowledge_skills': [col for col in competency_features if any(x in col for x in 
                                                       ['skill', 'education', 'work_year', 'capability'])],
        'social_role': [col for col in competency_features if any(x in col for x in 
                                                    ['industry', 'job_type'])],
        'self_concept': [col for col in competency_features if any(x in col for x in 
                                                     ['salary', 'location'])],
        'traits': [col for col in competency_features if any(x in col for x in 
                                                ['text_similarity', 'skill_text'])],
        'motive': [col for col in competency_features if any(x in col for x in 
                                                ['browse', 'exploration', 'focus'])]
    }
    
    # 训练模型
    print('训练满意度(satisfied)预测模型...')
    sub_sat, train_pred_sat, sat_feat_imp = sub_on_line(
        all_data[all_data['satisfied'] != -1], 
        all_data[all_data['satisfied'] == -1],
        use_feats, 'satisfied', ['live_city_id', 'city'], 
        use_cate=True
    )

    print('训练投递(delivered)预测模型...')
    sub_dev, train_pred_dev, dev_feat_imp = sub_on_line(
        all_data[all_data['delivered'] != -1], 
        all_data[all_data['delivered'] == -1],
        use_feats, 'delivered', ['live_city_id', 'city'], 
        use_cate=True
    )
    
    # 评估基础模型效果
    base_train_pred_sat = train_pred_sat.merge(
        all_data[all_data['satisfied'] != -1][['user_id', 'jd_no', 'delivered']],
        on=['user_id', 'jd_no'],
        how='left'
    )
    
    base_train_pred_sat = base_train_pred_sat.merge(
        train_pred_dev[['user_id', 'jd_no', 'delivered_pred']],
        on=['user_id', 'jd_no'],
        how='left'
    )
    
    # 检查必要列是否存在
    if check_columns_exist(base_train_pred_sat, ['delivered', 'delivered_pred', 'satisfied', 'satisfied_pred']):
        base_dev_map = offline_eval_map(base_train_pred_sat, 'delivered', 'delivered_pred')
        base_sat_map = offline_eval_map(base_train_pred_sat, 'satisfied', 'satisfied_pred')
        print('\n基础模型结果:')
        print('dev map:', round(base_dev_map, 4), 'sat map:', round(base_sat_map, 4), 'final score:',
            round(0.7 * base_sat_map + 0.3 * base_dev_map, 4))
    else:
        print("警告: 缺少评估必要的列，跳过基础模型评估")
    
    # 设计维度提升系数 - 根据先前分析重点提升特质和动机维度
    dim_boost = {
        'knowledge_skills': 1.0,
        'social_role': 0.8,
        'self_concept': 0.9, 
        'traits': 1.5,  # 特质维度重要性高，提升50%
        'motive': 1.3   # 动机维度次之，提升30%
    }
    
    # 为提高效率，直接使用一个合理的胜任力权重系数而非多次测试
    best_factor = 0.15  # 从先前试验中得到的优化值
    
    # 计算权重
    print(f"使用胜任力权重系数: {best_factor}")
    best_sat_weight, best_dev_weight, best_dim_weights, _, _ = calculate_adaptive_weights(
        sat_feat_imp, dev_feat_imp, competency_dimensions, best_factor, 0.7, dim_boost)
    
    # 合并胜任力维度特征到测试集预测结果
    print("计算最终预测结果...")
    sub_sat_comp = sub_sat.merge(
        all_data[all_data['satisfied'] == -1][['user_id', 'jd_no'] + 
                                            [f'comp_dim_{dim}' for dim in competency_dimensions.keys()] +
                                            ['competency_overall_score']],
        on=['user_id', 'jd_no'],
        how='left'
    )
    
    # 应用最佳权重计算最终预测得分
    sub_sat_comp['best_merge_prob'] = sub_sat['satisfied'] * best_sat_weight + \
                                    sub_dev['delivered'] * best_dev_weight
    
    # 添加各胜任力维度的加权分数
    for dim in competency_dimensions.keys():
        dim_col = f'comp_dim_{dim}'
        if dim_col in sub_sat_comp.columns:
            sub_sat_comp['best_merge_prob'] += sub_sat_comp[dim_col].fillna(0.5) * best_dim_weights[dim]
    
    # 生成最终结果 - 优化处理
    print("生成最终提交结果...")
    
    # 更高效的排序方法
    result_list = []
    
    # 使用向量化操作替代循环
    user_groups = sub_sat_comp.groupby('user_id')
    valid_jd_set = set(train_jd['jd_no'])
    
    for user_id, group in user_groups:
        # 将结果分为两部分：在训练集中的JD和不在训练集中的JD
        in_training = group[group['jd_no'].isin(valid_jd_set)].sort_values('best_merge_prob', ascending=False)
        not_in_training = group[~group['jd_no'].isin(valid_jd_set)]
        
        # 合并排序后的结果
        result_list.append(in_training[['user_id', 'jd_no', 'best_merge_prob']])
        result_list.append(not_in_training[['user_id', 'jd_no', 'best_merge_prob']])
    
    # 合并所有结果
    sub_df = pd.concat(result_list, ignore_index=True)
    
    # 保存最终结果
    sub_df[['user_id', 'jd_no']].to_csv('sub_optimized.csv', index=False)
    
    # 保存特征重要性分析结果
    sat_feat_imp[['Feature', 'avg_imp']].head(50).to_csv('feat_imp_satisfied_top50.csv', index=False, encoding='utf8')
    dev_feat_imp[['Feature', 'avg_imp']].head(50).to_csv('feat_imp_delivered_top50.csv', index=False, encoding='utf8')
    
    # 计算运行时间
    end_time = time.time()
    print(f"\n所有处理完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟")