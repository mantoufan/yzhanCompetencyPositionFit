# -*- coding：utf-8 -*-
# 改进的胜任力模型特征提取与预测 - 优化版

import pandas as pd
import numpy as np
import jieba as jb
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.spatial.distance import cosine
import re
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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
    tmp_file = open(jd_path, encoding='utf8')
    for i, j in enumerate(tmp_file.readlines()):
        if i == 175425:
            j = j.replace('销售\t|置业顾问\t|营销', '销售|置业顾问|营销')
        tmp = j.split('\t')
        tmp_list.append(tmp)
    tmp_file.close()
    return pd.DataFrame(tmp_list[1:], columns=tmp_list[0])

def get_min_salary(x):
    if len(x) == 12:
        return int(x[:6])
    elif len(x) == 10:
        return int(x[:5])
    elif len(x) == 11:
        return int(x[:5])
    elif len(x) == 9:
        return int(x[:4])
    else:
        return -1

def get_max_salary(x):
    if len(x) == 12:
        return int(x[6:])
    elif len(x) == 10:
        return int(x[5:])
    elif len(x) == 11:
        return int(x[5:])
    elif len(x) == 9:
        return int(x[4:])
    else:
        return -1

def is_same_user_city(df):
    live_city_id = str(df['live_city_id'])
    desire_jd_city = df['desire_jd_city_id']
    return live_city_id in desire_jd_city

def jieba_cnt(df):
    experience = df['experience']
    jd_title = df['jd_title']
    jd_sub_type = df['jd_sub_type']
    if isinstance(experience, str) and isinstance(jd_sub_type, str):
        tmp_set = set(jb.cut_for_search(jd_title)) | set(jb.cut_for_search(jd_sub_type))
        experience = set(jb.cut_for_search(experience))
        tmp_cnt = 0
        for t in tmp_set:
            if t in experience:
                tmp_cnt += 1
        return tmp_cnt
    else:
        return 0

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
    tfidf_enc_tmp = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(df[names])
    svd_tag_tmp = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = [f'{names}_svd_{i}' for i in range(10)]
    return pd.concat([df[[merge_id]], tag_svd_tmp], axis=1)

def get_str(x):
    return ' '.join([i for i in jb.cut(x) if i not in stop_words])

def offline_eval_map(train_df, label, pred_col):
    """
    增强版MAP评估函数，增加错误处理和日志
    """
    # 检查必要列
    if not all(col in train_df.columns for col in [label, pred_col]):
        print(f"错误: 评估需要的列 {label} 或 {pred_col} 不存在")
        return 0
    
    try:
        tmp_train = train_df.copy()
        tmp_train['rank'] = tmp_train.groupby('user_id')[pred_col].rank(ascending=False, method='first')
        tmp_x = tmp_train[tmp_train[label] == 1]
        
        if tmp_x.empty:
            print(f"警告: 没有找到标签 {label}=1 的样本")
            return 0
            
        tmp_x[f'{label}_index'] = tmp_x.groupby('user_id')['rank'].rank(ascending=True, method='first')
        tmp_x['score'] = tmp_x[f'{label}_index'] / tmp_train['rank']
        return tmp_x.groupby('user_id')['score'].mean().mean()
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return 0

# 优化：增强的文本相似度计算函数
def calculate_advanced_text_similarity(text1, text2, stop_words=None):
    """
    使用改进的TF-IDF和余弦相似度计算两段文本的语义相似度
    增加关键词匹配和词频权重
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0
    
    if len(text1) < 2 or len(text2) < 2:
        return 0
        
    if stop_words is None:
        stop_words = []
    
    # 分词处理 - 增加长度过滤和专业词汇识别
    text1_words = [w for w in jb.cut_for_search(text1) if w not in stop_words and len(w.strip()) > 1]
    text2_words = [w for w in jb.cut_for_search(text2) if w not in stop_words and len(w.strip()) > 1]
    
    if not text1_words or not text2_words:
        return 0
    
    # 计算Jaccard相似度
    set1, set2 = set(text1_words), set(text2_words)
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if set1 and set2 else 0
    
    # 构建词袋
    all_words = list(set(text1_words + text2_words))
    
    # 提取关键词 - 使用词频
    counter1 = Counter(text1_words)
    counter2 = Counter(text2_words)
    
    # 计算词频权重
    freq_weight = sum([min(counter1.get(w, 0), counter2.get(w, 0)) for w in all_words]) / max(len(text1_words), len(text2_words))
    
    # 计算TF-IDF相似度
    tfidf_similarity = 0
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([' '.join(text1_words), ' '.join(text2_words)])
        # 计算余弦相似度
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_similarity = 0
    
    # 综合多种相似度指标
    combined_similarity = (0.4 * jaccard + 0.4 * tfidf_similarity + 0.2 * freq_weight)
    
    return combined_similarity

# 优化：工作经验质量评估
def evaluate_experience_quality(experience, jd_title, jd_sub_type, stop_words=None):
    """
    评估工作经验的质量和相关性
    加入时间衰减和专业词权重
    """
    if not isinstance(experience, str) or not isinstance(jd_title, str):
        return 0
    
    # 解析经验段落，假设经验按|分隔
    exp_segments = experience.split('|') if '|' in experience else [experience]
    
    # 职位名称和类别
    job_keywords = set(jb.cut_for_search(jd_title)) | set(jb.cut_for_search(jd_sub_type))
    
    # 提取职位关键词 - 专业词汇识别
    key_words = []
    for word in job_keywords:
        if len(word) >= 2 and not any(c.isdigit() for c in word):
            key_words.append(word)
    
    # 如果没有有效关键词，使用所有词
    if not key_words:
        key_words = list(job_keywords)
    
    # 计算每段经验的相关性并加权
    total_score = 0
    weights = []
    exp_scores = []
    
    # 计算衰减权重系数 - 考虑经验的时间衰减，最近经验权重更高
    decay_factor = 0.8
    
    # 越近期的经验权重越高
    for i, segment in enumerate(exp_segments):
        # 指数衰减权重
        weight = decay_factor ** i  
        weights.append(weight)
        
        # 计算关键词匹配
        segment_words = set(jb.cut_for_search(segment))
        if not segment_words:
            exp_scores.append(0)
            continue
            
        # 计算关键词匹配度 - 使用TF-IDF权重
        matches = len(job_keywords.intersection(segment_words))
        
        # 计算关键词匹配率，考虑文本长度的影响
        matching_ratio = matches / math.sqrt(len(segment_words))
        
        # 检查是否包含特别重要的关键词
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

# 优化：教育背景评分函数
def score_education_match(cur_degree_num, required_degree_num, jd_title=None):
    """
    计算教育背景匹配度的连续评分
    加入专业与职位的相关性判断
    """
    if np.isnan(cur_degree_num) or np.isnan(required_degree_num):
        return 0.5  # 返回中性值而非0
        
    # 如果达到要求，基础分为1.0
    if cur_degree_num >= required_degree_num:
        base_score = 1.0
        # 超过要求的部分有额外加分，但有上限
        degree_diff = cur_degree_num - required_degree_num
        # 使用对数函数平滑超额学历的奖励
        extra_score = min(math.log1p(degree_diff) * 0.2, 0.4)
        return base_score + extra_score
    else:
        # 未达到要求，根据差距给予部分分数 - 使用指数衰减
        gap = required_degree_num - cur_degree_num
        return max(0.2, math.exp(-gap * 0.5))  # 最低分为0.2而非0

# 优化：工作年限匹配评分函数
def score_work_year_match(user_work_year, min_work_year, max_work_year=None):
    """
    计算工作年限匹配度的连续评分
    使用平滑的sigmoid函数替代简单阈值
    """
    if np.isnan(user_work_year) or np.isnan(min_work_year):
        return 0.5  # 中性值
    
    # 如果没有明确的最大年限要求，设定一个合理的上限
    if max_work_year is None or np.isnan(max_work_year):
        max_work_year = min_work_year * 2.5
    
    # 计算标准化的工作年限比例
    if min_work_year > 0:
        # 使用sigmoid函数平滑过渡
        min_ratio = 1 / (1 + math.exp(-2 * (user_work_year / min_work_year - 1)))
        
        # 如果超过最大年限，计算惩罚系数
        if user_work_year > max_work_year:
            over_ratio = (user_work_year - max_work_year) / max_work_year
            # 使用柔和的惩罚函数
            penalty = 1 - 0.2 * min(1, math.tanh(over_ratio))
            return min_ratio * penalty
        else:
            return min_ratio
    else:
        # 无最低要求时，只要不超过最大值即可
        if max_work_year > 0 and user_work_year > max_work_year:
            over_ratio = (user_work_year - max_work_year) / max_work_year
            return 1 - 0.2 * min(1, math.tanh(over_ratio))
        else:
            return 0.8  # 基础匹配分

# 优化：薪资匹配评分函数
def score_salary_match(min_desire, max_desire, min_offer, max_offer):
    """
    计算期望薪资与岗位薪资的匹配度
    使用重叠率和满足率综合评分
    """
    if min_desire <= 0 or min_offer <= 0:
        return 0.5  # 数据不足时返回中性值
    
    # 计算薪资范围
    desire_range = max_desire - min_desire if max_desire > min_desire else min_desire * 0.5
    offer_range = max_offer - min_offer if max_offer > min_offer else min_offer * 0.5
    
    # 确保范围有效
    desire_range = max(desire_range, min_desire * 0.1)
    offer_range = max(offer_range, min_offer * 0.1)
    
    # 计算区间重叠
    lower_bound = max(min_desire, min_offer)
    upper_bound = min(max_desire if max_desire > 0 else min_desire * 1.5, 
                     max_offer if max_offer > 0 else min_offer * 1.5)
    
    overlap = max(0, upper_bound - lower_bound)
    
    # 计算重叠率
    overlap_ratio = overlap / min(desire_range, offer_range)
    
    # 计算薪资满足率 - 岗位薪资满足期望的程度
    if min_offer >= min_desire and (max_offer >= max_desire or max_desire <= 0):
        # 完全满足
        satisfaction = 1.0
    elif min_offer >= min_desire:
        # 部分满足
        if max_desire > 0:
            satisfaction = 0.7 + 0.3 * min(1, max_offer / max_desire)
        else:
            satisfaction = 0.8
    else:
        # 最低薪资不满足期望
        satisfaction = min(0.6, (min_offer / min_desire) * 0.7)
    
    # 加权计算最终得分
    final_score = 0.6 * satisfaction + 0.4 * overlap_ratio
    
    # 奖励高薪溢价
    if min_offer > min_desire * 1.2:
        premium = min(0.2, math.log(min_offer / min_desire, 2) * 0.1)
        final_score += premium
    
    return min(1.0, final_score)

# 优化：地域匹配函数
def score_location_match(live_city, desire_cities, job_city):
    """
    计算地域匹配度的连续得分
    考虑地域偏好强度
    """
    # 确保数据可用
    if not isinstance(live_city, (int, str)) or not isinstance(job_city, (int, str)):
        return 0.5
    
    live_city = str(live_city)
    job_city = str(job_city)
    
    # 转换desire_cities为列表格式
    if isinstance(desire_cities, list):
        desire_cities = [str(city) for city in desire_cities]
    else:
        desire_cities = []
    
    # 计算地域偏好强度 - 期望城市越少，偏好越强
    preference_strength = max(0.1, min(1.0, 1 / math.sqrt(max(1, len(desire_cities)))))
    
    # 基础匹配评分
    if live_city == job_city:
        # 当前居住地与工作地相同，最高评分
        return 1.0
    elif job_city in desire_cities:
        # 工作地在期望地点列表中，按偏好强度给分
        return 0.7 + 0.3 * preference_strength
    else:
        # 工作地既不是当前居住地，也不在期望列表中
        if desire_cities:
            # 有明确地域偏好但不匹配，按偏好强度惩罚
            return 0.5 - 0.3 * preference_strength
        else:
            # 没有明确地域偏好，返回中性值
            return 0.6  # 略高于中性，因为无偏好意味着更灵活

# 优化：行业匹配函数
def score_industry_match(cur_industry, desire_industries, jd_industry=None):
    """
    计算行业匹配度的连续评分
    考虑跨行业能力迁移
    """
    if not isinstance(cur_industry, str) or not isinstance(desire_industries, str):
        return 0.5
    
    # 分解行业ID
    cur_industries = cur_industry.split(',') if ',' in cur_industry else [cur_industry]
    desire_industries = desire_industries.split(',') if ',' in desire_industries else [desire_industries]
    
    # 提取行业大类
    cur_categories = [ind[:2] for ind in cur_industries if len(ind) >= 2]
    desire_categories = [ind[:2] for ind in desire_industries if len(ind) >= 2]
    
    # 计算行业偏好多样性
    diversity = len(set(desire_categories)) / len(desire_categories) if desire_categories else 0.5
    
    # 精确匹配检查
    exact_matches = set(cur_industries).intersection(set(desire_industries))
    if exact_matches:
        # 精确匹配权重受多样性影响
        match_score = 0.8 + 0.2 * (1 - diversity)  # 多样性低时加分更高
    elif set(cur_categories).intersection(set(desire_categories)):
        # 大类匹配 - 有能力迁移潜力
        match_score = 0.6 + 0.2 * (1 - diversity)
    else:
        # 无匹配，但考虑行业经验的广度
        match_score = 0.3 + 0.1 * len(cur_categories) / max(1, len(desire_categories))
    
    # 如果有岗位行业信息，进一步评估
    if isinstance(jd_industry, str):
        jd_industries = jd_industry.split(',') if ',' in jd_industry else [jd_industry]
        jd_categories = [ind[:2] for ind in jd_industries if len(ind) >= 2]
        
        # 岗位行业与期望匹配
        if set(jd_industries).intersection(set(desire_industries)):
            match_score += 0.15
        elif set(jd_categories).intersection(set(desire_categories)):
            match_score += 0.1
        
        # 岗位行业与当前行业匹配
        if set(jd_industries).intersection(set(cur_industries)):
            match_score += 0.15
        elif set(jd_categories).intersection(set(cur_categories)):
            match_score += 0.1
    
    return min(1.0, match_score)

# 优化：动机和行为特征评估
def evaluate_motivation_and_behavior(df):
    """
    评估用户的动机和浏览行为特征
    增加行为模式分析
    """
    features = {}
    
    # 浏览强度 - 使用分段函数
    if isinstance(df['user_jd_cnt'], (int, float)) and not np.isnan(df['user_jd_cnt']):
        # 使用分段函数处理浏览强度
        browse_cnt = df['user_jd_cnt']
        if browse_cnt <= 2:  # 轻度浏览
            features['browse_intensity'] = 0.3 + 0.2 * browse_cnt
        elif browse_cnt <= 5:  # 中度浏览
            features['browse_intensity'] = 0.5 + 0.1 * (browse_cnt - 2)
        else:  # 深度浏览
            features['browse_intensity'] = 0.8 + 0.2 * min(1, math.log(browse_cnt / 5))
    else:
        features['browse_intensity'] = 0.3  # 低于平均值
    
    # 职位探索多样性 - 考虑相对比例和绝对多样性
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['jd_nunique'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['jd_nunique']) and df['jd_nunique'] > 0:
        
        # 相对多样性 - 不同类型占总浏览的比例
        diversity_ratio = df['jd_sub_type_nunique'] / df['jd_nunique']
        
        # 绝对多样性 - 浏览的不同类型数量
        absolute_diversity = min(1.0, df['jd_sub_type_nunique'] / 10)
        
        # 综合考虑相对和绝对多样性
        features['job_exploration_diversity'] = 0.7 * diversity_ratio + 0.3 * absolute_diversity
    else:
        features['job_exploration_diversity'] = 0.4  # 默认值略低于平均
    
    # 浏览稳定性 - 考虑浏览间隔的规律性
    if 'jd_days_std' in df and isinstance(df['jd_days_std'], (int, float)) and not np.isnan(df['jd_days_std']):
        # 标准差越小，表示浏览越规律
        stability = 1.0 / (1.0 + df['jd_days_std'] / max(1, df['jd_days_mean'] if 'jd_days_mean' in df else 1))
        features['browse_stability'] = stability
    else:
        features['browse_stability'] = 0.5  # 默认中性值
    
    # 行为专注度 - 新增特征
    if isinstance(df['jd_sub_type_nunique'], (int, float)) and isinstance(df['user_jd_cnt'], (int, float)) and \
       not np.isnan(df['jd_sub_type_nunique']) and not np.isnan(df['user_jd_cnt']) and df['user_jd_cnt'] > 0:
        
        # 计算职位类型的重复浏览率
        focus_ratio = (df['user_jd_cnt'] - df['jd_sub_type_nunique']) / df['user_jd_cnt']
        features['browse_focus'] = focus_ratio
    else:
        features['browse_focus'] = 0.4  # 默认值
    
    return features

# 优化：提取胜任力维度特征的函数
def extract_competency_dimensions(df, stop_words=None):
    """
    提取增强的胜任力各维度特征
    基于Spencer & Spencer胜任力冰山模型，加入更多连续评分和交互特征
    """
    features = {}
    
    # 1. 知识和技能维度特征 - 改进为连续评分
    # 技能匹配 - 使用改进的计算方法
    if isinstance(df['experience'], str) and isinstance(df['jd_title'], str) and isinstance(df['jd_sub_type'], str):
        features['skill_match'] = evaluate_experience_quality(df['experience'], df['jd_title'], df['jd_sub_type'], stop_words)
    else:
        features['skill_match'] = 0.4  # 默认值设为中等偏低
    
    # 教育背景 - 使用连续评分函数
    if 'cur_degree_id_num' in df and not np.isnan(df['cur_degree_id_num']):
        min_edu_num = df['min_edu_level_num'] if not np.isnan(df['min_edu_level_num']) else 3  # 默认要求大专
        features['education_match'] = score_education_match(df['cur_degree_id_num'], min_edu_num, df['jd_title'] if 'jd_title' in df else None)
    else:
        features['education_match'] = 0.5  # 中性值
    
    # 工作年限 - 使用连续评分函数
    if isinstance(df['user_work_year'], (int, float)) and not np.isnan(df['user_work_year']):
        min_year = df['min_work_year'] if not np.isnan(df['min_work_year']) else 1
        max_year = df['max_work_year'] if 'max_work_year' in df and not np.isnan(df['max_work_year']) else None
        features['work_year_match'] = score_work_year_match(df['user_work_year'], min_year, max_year)
    else:
        features['work_year_match'] = 0.5  # 中性值
    
    # 2. 社会角色维度特征 - 改进为连续评分
    # 行业匹配
    if isinstance(df['cur_industry_id'], str) and isinstance(df['desire_jd_industry_id'], str):
        jd_industry = df['jd_industry'] if 'jd_industry' in df and isinstance(df['jd_industry'], str) else None
        features['industry_match'] = score_industry_match(df['cur_industry_id'], df['desire_jd_industry_id'], jd_industry)
    else:
        features['industry_match'] = 0.5  # 中性值
    
    # 职位类型匹配 - 添加职位层级考虑
    if isinstance(df['desire_jd_type_id'], str) and isinstance(df['jd_sub_type'], str):
        direct_match = df['jd_sub_type'] in df['desire_jd_type_id']
        
        if direct_match:
            features['job_type_match'] = 1.0
        else:
            # 检查大类匹配
            jd_types = df['desire_jd_type_id'].split(',') if ',' in df['desire_jd_type_id'] else [df['desire_jd_type_id']]
            if any(df['jd_sub_type'].startswith(jt[:2]) for jt in jd_types if len(jt) >= 2):
                features['job_type_match'] = 0.7
            else:
                features['job_type_match'] = 0.3
    else:
        features['job_type_match'] = 0.5  # 中性值
    
    # 3. 自我概念维度特征 - 改进为连续评分
    # 薪资期望匹配
    if isinstance(df['min_desire_salary'], (int, float)) and isinstance(df['min_salary'], (int, float)) and \
       not np.isnan(df['min_desire_salary']) and not np.isnan(df['min_salary']):
        max_desire = df['max_desire_salary'] if 'max_desire_salary' in df and \
                    isinstance(df['max_desire_salary'], (int, float)) and not np.isnan(df['max_desire_salary']) else 0
        max_offer = df['max_salary'] if 'max_salary' in df and \
                   isinstance(df['max_salary'], (int, float)) and not np.isnan(df['max_salary']) else 0
        features['salary_match'] = score_salary_match(df['min_desire_salary'], max_desire, df['min_salary'], max_offer)
    else:
        features['salary_match'] = 0.5  # 中性值
    
    # 地点偏好匹配
    features['location_match'] = score_location_match(
        df['live_city_id'], 
        df['desire_jd_city_id'] if 'desire_jd_city_id' in df else [], 
        df['city']
    )
    
    # 4. 特质维度特征 - 改进文本相似度
    # 职位描述与经验的高级文本相似度
    job_desc = df['job_description\n'] if isinstance(df['job_description\n'], str) else ""
    experience = df['experience'] if isinstance(df['experience'], str) else ""
    
    # 使用改进的文本相似度函数
    features['text_similarity'] = calculate_advanced_text_similarity(job_desc, experience, stop_words)
    
    # 5. 动机维度特征 - 添加更多行为特征
    motivation_features = evaluate_motivation_and_behavior(df)
    features.update(motivation_features)
    
    # 6. 特征交互
    # 添加重要特征的交互项
    if 'skill_match' in features and 'text_similarity' in features:
        # 非线性交互 - 使用加权几何平均而非简单乘积
        weight1, weight2 = 0.6, 0.4  # 技能匹配权重更高
        if features['skill_match'] > 0 and features['text_similarity'] > 0:
            features['skill_text_interaction'] = (features['skill_match'] ** weight1) * (features['text_similarity'] ** weight2)
        else:
            features['skill_text_interaction'] = 0
    
    if 'education_match' in features and 'work_year_match' in features:
        # 使用几何平均加权综合
        edu_weight = 0.4
        work_weight = 0.6
        features['edu_exp_interaction'] = (features['education_match'] ** edu_weight) * (features['work_year_match'] ** work_weight)
    
    if 'salary_match' in features and 'skill_match' in features:
        # 技能与薪资的匹配比例，反映"值不值这个价"
        if features['skill_match'] > 0.2:  # 有基本技能匹配才有意义
            # 使用对数比率避免极端值
            ratio = features['salary_match'] / features['skill_match']
            features['salary_skill_ratio'] = 0.5 + 0.5 * math.tanh((ratio - 1) * 2)  # 归一化到0.5附近
        else:
            features['salary_skill_ratio'] = 0.4  # 技能不匹配时略低于中位值
    
    # 7. 职业发展匹配度 - 新增特征
    if 'work_year_match' in features and 'job_type_match' in features and 'industry_match' in features:
        # 综合评估职业发展的匹配程度
        features['career_path_match'] = 0.4 * features['work_year_match'] + 0.3 * features['job_type_match'] + 0.3 * features['industry_match']
    
    # 8. 综合能力匹配 - 新增特征
    if 'skill_match' in features and 'education_match' in features and 'work_year_match' in features:
        features['capability_match'] = 0.5 * features['skill_match'] + 0.2 * features['education_match'] + 0.3 * features['work_year_match']
    
    # 归一化特征值到0-1范围
    for key in features:
        if features[key] > 1.0:
            features[key] = 1.0
        elif features[key] < 0:
            features[key] = 0.0
    
    return features

# 优化：为数据集添加胜任力维度特征的函数
def add_competency_dimension_features(df, stop_words=None):
    """为数据集添加胜任力维度特征，包括标准化处理"""
    print("提取胜任力维度特征...")
    start_time = time.time()
    
    # 提取胜任力维度特征
    competency_features = df.apply(lambda x: extract_competency_dimensions(x, stop_words), axis=1)
    
    # 获取所有特征名称
    all_feature_names = set()
    for features in competency_features:
        all_feature_names.update(features.keys())
    
    # 将特征添加到数据框
    for feature_name in all_feature_names:
        df[f'comp_{feature_name}'] = competency_features.apply(lambda x: x.get(feature_name, 0.5))  # 默认值为0.5而非0
    
    # 标准化处理 - 优化为稳健标准化
    print("标准化胜任力特征...")
    valid_comp_cols = [f'comp_{feature_name}' for feature_name in all_feature_names]
    
    # 按维度进行标准化处理 - 而非全局标准化，保持各维度的独立性
    dim_mapping = {
        'knowledge_skills': ['skill_match', 'education_match', 'work_year_match', 'edu_exp_interaction', 'capability_match'],
        'social_role': ['industry_match', 'job_type_match', 'career_path_match'],
        'self_concept': ['salary_match', 'location_match', 'salary_skill_ratio'],
        'traits': ['text_similarity', 'skill_text_interaction'],
        'motive': ['browse_intensity', 'job_exploration_diversity', 'browse_stability', 'browse_focus']
    }
    
    # 为每个维度分别标准化
    for dim, features in dim_mapping.items():
        dim_cols = [f'comp_{f}' for f in features if f'comp_{f}' in df.columns]
        if dim_cols:
            # 首先使用分位数归一化避免异常值影响
            df[dim_cols] = df[dim_cols].rank(pct=True).clip(0.05, 0.95)
            
            # 然后应用MinMaxScaler使得分布更均匀
            scaler = MinMaxScaler()
            df[dim_cols] = scaler.fit_transform(df[dim_cols].fillna(0.5))
    
    # 为每个维度计算综合得分 - 使用加权平均而非简单平均
    weights = {
        'knowledge_skills': {'skill_match': 0.5, 'education_match': 0.15, 'work_year_match': 0.2, 
                           'edu_exp_interaction': 0.05, 'capability_match': 0.1},
        'social_role': {'industry_match': 0.4, 'job_type_match': 0.4, 'career_path_match': 0.2},
        'self_concept': {'salary_match': 0.4, 'location_match': 0.4, 'salary_skill_ratio': 0.2},
        'traits': {'text_similarity': 0.4, 'skill_text_interaction': 0.6},
        'motive': {'browse_intensity': 0.3, 'job_exploration_diversity': 0.3, 
                 'browse_stability': 0.2, 'browse_focus': 0.2}
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
            # 归一化权重确保和为1
            norm_weights = [w/sum(dim_weights) for w in dim_weights]
            
            # 计算加权平均
            df[f'comp_dim_{dim}'] = 0
            for i, col in enumerate(dim_cols):
                df[f'comp_dim_{dim}'] += df[col] * norm_weights[i]
    
    # 计算整体胜任力得分 - 使用组合加权方法
    # 按维度加权，特质和动机维度权重高于其他
    dim_importance = {
        'knowledge_skills': 0.25,
        'social_role': 0.15,
        'self_concept': 0.15,
        'traits': 0.25, 
        'motive': 0.20
    }
    
    dim_cols = [f'comp_dim_{dim}' for dim in dim_importance if f'comp_dim_{dim}' in df.columns]
    if dim_cols:
        df['competency_overall_score'] = 0
        for dim, weight in dim_importance.items():
            col = f'comp_dim_{dim}'
            if col in df.columns:
                df['competency_overall_score'] += df[col] * weight
    
    # 添加胜任力"协同加成"特征 - 多维度高分时的奖励
    if 'comp_dim_traits' in df.columns and 'comp_dim_knowledge_skills' in df.columns:
        # 技能-特质协同 - 当两个维度都高时，整体表现更好
        skill_trait_synergy = df['comp_dim_traits'] * df['comp_dim_knowledge_skills']
        # 使用sigmoid函数强化高分对的效果
        df['comp_skill_trait_synergy'] = 1 / (1 + np.exp(-5 * (skill_trait_synergy - 0.5)))
    
    if 'comp_dim_motive' in df.columns and 'comp_dim_self_concept' in df.columns:
        # 动机-自我概念协同 - 反映个人职业偏好与岗位匹配
        df['comp_motive_self_synergy'] = df['comp_dim_motive'] * df['comp_dim_self_concept']
    
    print(f"胜任力特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    return df

# 优化：分析胜任力特征重要性的函数
def analyze_feature_importance(imp_df, competency_features, top_n=20):
    """详细分析胜任力特征的重要性和维度分布"""
    # 只保留有实际重要性的特征
    imp_df = imp_df[imp_df['avg_imp'] > 0].copy()
    
    competency_imp = imp_df[imp_df['Feature'].isin(competency_features)].copy()
    total_comp_imp = competency_imp['avg_imp'].sum()
    total_imp = imp_df['avg_imp'].sum()
    
    print(f"胜任力特征总重要性: {total_comp_imp:.4f}, 占比: {total_comp_imp/total_imp:.2%}")
    
    # 按维度分组分析
    dim_mapping = {
        '知识技能维度': ['comp_skill_match', 'comp_education_match', 'comp_work_year_match', 
                    'comp_edu_exp_interaction', 'comp_capability_match'],
        '社会角色维度': ['comp_industry_match', 'comp_job_type_match', 'comp_career_path_match'],
        '自我概念维度': ['comp_salary_match', 'comp_location_match', 'comp_salary_skill_ratio'],
        '特质维度': ['comp_text_similarity', 'comp_skill_text_interaction', 'comp_skill_trait_synergy'],
        '动机维度': ['comp_browse_intensity', 'comp_job_exploration_diversity', 
                 'comp_browse_stability', 'comp_browse_focus', 'comp_motive_self_synergy']
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
            
            # 显示该维度内的特征重要性
            dim_imp_df = competency_imp[competency_imp['Feature'].isin(dim_features)].sort_values('avg_imp', ascending=False)
            for _, row in dim_imp_df.iterrows():
                print(f"    - {row['Feature']}: {row['avg_imp']:.4f}, 占比: {row['avg_imp']/total_imp:.2%}")
    
    # 显示所有胜任力特征中最重要的N个
    print(f"\n前{top_n}个最重要的胜任力特征:")
    top_comp_features = competency_imp.sort_values('avg_imp', ascending=False).head(top_n)
    for _, row in top_comp_features.iterrows():
        print(f"  {row['Feature']}: {row['avg_imp']:.4f}, 占比: {row['avg_imp']/total_imp:.2%}")
    
    return competency_imp, dim_importances

# 优化：模型训练和预测函数
def sub_on_line(train_, test_, pred, label, cate_cols, is_shuffle=True, use_cate=True):
    print(f'data shape:\ntrain--{train_.shape}\ntest--{test_.shape}')
    n_splits = 5
    
    # 使用分层KFold代替普通KFold，确保每个折的标签分布一致
    if is_shuffle:
        # 为每个用户分配一个组，确保同一用户的数据不会分散到不同折
        user_groups = {}
        user_ids = train_['user_id'].unique()
        for i, uid in enumerate(user_ids):
            user_groups[uid] = i % n_splits
        
        # 创建用户分组索引
        train_folds = []
        for fold in range(n_splits):
            train_idx = []
            valid_idx = []
            for i, uid in enumerate(user_ids):
                if user_groups[uid] == fold:
                    valid_idx.append(i)
                else:
                    train_idx.append(i)
            train_folds.append((train_idx, valid_idx))
    else:
        # 使用标准KFold
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)
        train_folds = list(folds.split(train_['user_id'].unique()))
    
    sub_preds = np.zeros((test_.shape[0], n_splits))
    train_[f'{label}_pred'] = 0
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = pred
    
    print(f'Use {len(pred)} features ...')
    auc_scores = []
    
    # 优化LightGBM参数 - 保持单轮迭代但优化其他参数
    params = {
        'learning_rate': 0.1,  # 提高学习率，因为只使用一轮迭代
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 127,  # 增加叶子数以提高模型复杂度
        'max_depth': 7,  # 控制树的深度
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,  # 减少最小叶子样本数
        'lambda_l1': 0.5,  # 添加L1正则化
        'lambda_l2': 0.5,  # 添加L2正则化
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'nthread': -1,
        'verbose': -1
    }
    
    train_user_id = train_['user_id'].unique()
    for n_fold, (train_idx, valid_idx) in enumerate(train_folds, start=1):
        print(f'the {n_fold} training start ...')
        train_x, train_y = train_.loc[train_['user_id'].isin(train_user_id[train_idx]), pred], train_.loc[
            train_['user_id'].isin(train_user_id[train_idx]), label]
        valid_x, valid_y = train_.loc[train_['user_id'].isin(train_user_id[valid_idx]), pred], train_.loc[
            train_['user_id'].isin(train_user_id[valid_idx]), label]
        
        print(f'for train user:{len(train_idx)}\nfor valid user:{len(valid_idx)}')
        
        # 处理类别特征
        if use_cate:
            # 检查哪些分类特征实际存在于数据中
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
        
        # 预测并保存结果
        sub_preds[:, n_fold - 1] = clf.predict(test_[pred], num_iteration=clf.best_iteration)
        auc_scores.append(clf.best_score['valid_0']['auc'])
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance()
        train_.loc[train_['user_id'].isin(train_user_id[valid_idx]), f'{label}_pred'] = \
            clf.predict(valid_x, num_iteration=clf.best_iteration)

    # 计算平均特征重要性
    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    
    # 保存详细特征重要性
    fold_importance_df[['Feature', 'avg_imp']].to_csv(f'feat_imp_details_{label}.csv', index=False, encoding='utf8')
    
    # 预测值使用加权平均而非简单平均 - 加权方法根据每个折的AUC表现
    auc_weights = np.array(auc_scores) / sum(auc_scores)
    test_[label] = np.sum(sub_preds * auc_weights.reshape(1, -1), axis=1)
    
    print('auc score', np.mean(auc_scores))
    return test_[['user_id', 'jd_no', label]], train_[['user_id', 'jd_no', f'{label}_pred', label]], fold_importance_df

# 优化：自适应维度权重计算
def calculate_adaptive_weights(sat_feat_imp, dev_feat_imp, competency_dimensions, 
                             balance_factor=0.2, sat_ratio=0.7, dim_boost=None):
    """
    计算各维度的自适应权重
    支持单独调整维度权重的提升系数
    """
    # 提取胜任力特征
    competency_features = []
    for dim, features in competency_dimensions.items():
        competency_features.extend(features)
    
    # 计算满意度预测和投递预测中的胜任力特征重要性
    comp_sat_imp, sat_dim_importances = analyze_feature_importance(sat_feat_imp, competency_features, top_n=10)
    comp_dev_imp, dev_dim_importances = analyze_feature_importance(dev_feat_imp, competency_features, top_n=10)
    
    # 计算总重要性
    sat_total_imp = sat_feat_imp['avg_imp'].sum()
    dev_total_imp = dev_feat_imp['avg_imp'].sum()
    
    # 计算各维度在总体中的权重
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
    
    # 计算满意度预测中各维度权重
    for dim in competency_dimensions:
        dim_chn = dim_name_map.get(dim, dim)
        dim_imp = sat_dim_importances.get(dim_chn, 0)
        dim_weights_sat[dim] = dim_imp / sat_total_imp if sat_total_imp > 0 else 0
    
    # 计算投递预测中各维度权重
    for dim in competency_dimensions:
        dim_chn = dim_name_map.get(dim, dim)
        dim_imp = dev_dim_importances.get(dim_chn, 0)
        dim_weights_dev[dim] = dim_imp / dev_total_imp if dev_total_imp > 0 else 0
    
    # 总体胜任力权重
    comp_sat_total_weight = sum(dim_weights_sat.values())
    comp_dev_total_weight = sum(dim_weights_dev.values())
    
    # 平衡参数 - 控制胜任力模型与基础模型的权重比例
    if dim_boost is None:
        dim_boost = {dim: 1.0 for dim in competency_dimensions}
    
    # 计算最终预测的基础权重
    sat_weight = (1 - balance_factor) * sat_ratio
    dev_weight = (1 - balance_factor) * (1 - sat_ratio)
    
    # 各维度最终权重
    final_dim_weights = {}
    for dim in competency_dimensions:
        # 各维度在满意度和投递预测中的分配权重
        sat_dim_part = dim_weights_sat.get(dim, 0) / max(comp_sat_total_weight, 0.001) * balance_factor * sat_ratio
        dev_dim_part = dim_weights_dev.get(dim, 0) / max(comp_dev_total_weight, 0.001) * balance_factor * (1 - sat_ratio)
        
        # 应用维度提升系数
        boost = dim_boost.get(dim, 1.0)
        final_dim_weights[dim] = (sat_dim_part + dev_dim_part) * boost
    
    # 输出权重分配情况
    print("\n最终模型权重分配:")
    print(f"满意度预测基础权重: {sat_weight:.4f}")
    print(f"投递预测基础权重: {dev_weight:.4f}")
    print("胜任力维度权重:")
    for dim, weight in final_dim_weights.items():
        print(f"  - {dim}: {weight:.4f} (提升系数: {dim_boost.get(dim, 1.0):.2f})")
    
    return sat_weight, dev_weight, final_dim_weights, comp_sat_imp, comp_dev_imp

if __name__ == "__main__":
    min_work_year = {103: 1, 305: 3, 510: 5, 1099: 10}
    max_work_year = {103: 3, 305: 5, 510: 10}
    degree_map = {'其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
                  '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}

    sub_path = './submit/'
    train_data_path = './train/'
    test_data_path = './test/'
    
    # 数据加载和预处理部分
    train_user = pd.read_csv(train_data_path + 'table1_user.csv', sep=',')
    train_user['desire_jd_city_id'] = train_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    train_user['desire_jd_salary_id'] = train_user['desire_jd_salary_id'].astype(str)
    train_user['min_desire_salary'] = train_user['desire_jd_salary_id'].apply(get_min_salary)
    train_user['max_desire_salary'] = train_user['desire_jd_salary_id'].apply(get_max_salary)
    train_user['min_cur_salary'] = train_user['cur_salary_id'].apply(get_min_salary)
    train_user['max_cur_salary'] = train_user['cur_salary_id'].apply(get_max_salary)
    train_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)
    train_jd = pd.read_csv(train_data_path + 'table2_jd.csv', sep='\t')
    train_jd.drop(['company_name', 'max_edu_level', 'is_mangerial', 'resume_language_required'], axis=1, inplace=True)

    train_jd['min_work_year'] = train_jd['min_years'].map(min_work_year)
    train_jd['max_work_year'] = train_jd['min_years'].map(max_work_year)
    train_jd['start_date'].replace(r'\N', '22000101', inplace=True)
    train_jd['end_date'].replace(r'\N', '22000101', inplace=True)
    train_jd['start_date'] = pd.to_datetime(train_jd['start_date'].astype(str).apply(lambda x:
                                                                                     f'{x[:4]}-{x[4:6]}-{x[6:]}'))
    train_jd['end_date'] = pd.to_datetime(train_jd['end_date'].astype(str).apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:]}'))
    train_jd.loc[train_jd['end_date'] == '2200-01-01', ['start_date', 'end_date']] = np.nan

    print('加载停用词列表...')
    stop_words = [i.strip() for i in open('中文停用词表.txt', 'r', encoding='utf8').readlines()]
    stop_words.extend(['\n', '\xa0', '\u3000', '\u2002'])
    
    tmp_cut = Parallel(n_jobs=-1)(delayed(get_str)(train_jd.loc[ind]['job_description\n'])
                                  for ind in tqdm(train_jd.index))

    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(tmp_cut)
    svd_tag = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    tag_svd = svd_tag.fit_transform(tfidf_vec)
    tag_svd = pd.DataFrame(tag_svd)
    tag_svd.columns = [f'desc_svd_{i}' for i in range(10)]
    train_jd = pd.concat([train_jd, tag_svd], axis=1)

    train_action = pd.read_csv(train_data_path + 'table3_action.csv', sep=',')
    train_action['user_jd_cnt'] = train_action.groupby(['user_id', 'jd_no'])['jd_no'].transform('count').values
    train_action['jd_cnt'] = train_action.groupby(['user_id'])['jd_no'].transform('count').values
    train_action['jd_nunique'] = train_action.groupby(['user_id'])['jd_no'].transform('nunique').values
    train_action = train_action.drop_duplicates()
    train_action.sort_values(['user_id', 'jd_no', 'delivered', 'satisfied'], inplace=True)
    train_action = train_action.drop_duplicates(subset=['user_id', 'jd_no'], keep='last')
    train_action = train_action[train_action['jd_no'].isin(train_jd['jd_no'].unique())]

    train = train_action.merge(train_user, on='user_id', how='left')
    train = train.merge(train_jd, on='jd_no', how='left')
    del train['browsed']

    print('train data base feats already generated ...')

    test_user = pd.read_csv(test_data_path + 'user_ToBePredicted.csv', sep='\t')
    test_user['desire_jd_city_id'] = test_user['desire_jd_city_id'].apply(lambda x: re.findall('\d+', x))
    test_user['desire_jd_salary_id'] = test_user['desire_jd_salary_id'].astype(str)
    test_user['min_desire_salary'] = test_user['desire_jd_salary_id'].apply(get_min_salary)
    test_user['max_desire_salary'] = test_user['desire_jd_salary_id'].apply(get_max_salary)
    test_user['min_cur_salary'] = test_user['cur_salary_id'].apply(get_min_salary)
    test_user['max_cur_salary'] = test_user['cur_salary_id'].apply(get_max_salary)
    test_user.drop(['desire_jd_salary_id', 'cur_salary_id'], axis=1, inplace=True)

    test = pd.read_csv(test_data_path + 'zhaopin_round1_user_exposure_B_20190819.csv', sep=' ')
    test['user_jd_cnt'] = test.groupby(['user_id', 'jd_no'])['jd_no'].transform('count').values
    test['jd_cnt'] = test.groupby(['user_id'])['jd_no'].transform('count').values
    test['jd_nunique'] = test.groupby(['user_id'])['jd_no'].transform('nunique').values
    test = test.drop_duplicates()

    test['delivered'] = -1
    test['satisfied'] = -1

    test = test.merge(test_user, on='user_id', how='left')
    test = test.merge(train_jd, on='jd_no', how='left')

    print('test data base feats already generated ...')

    all_data = pd.concat([train, test], ignore_index=True)

    all_data['jd_user_cnt'] = all_data.groupby(['jd_no'])['user_id'].transform('count').values
    all_data['same_user_city'] = all_data.apply(is_same_user_city, axis=1).astype(int)
    all_data['city'].fillna(-1, inplace=True)
    all_data['city'] = all_data['city'].astype(int)
    all_data['same_com_live'] = (all_data['city'] == all_data['live_city_id']).astype(int)
    all_data['min_edu_level'] = all_data['min_edu_level'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    all_data['cur_degree_id'] = all_data['cur_degree_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    all_data['min_edu_level_num'] = all_data['min_edu_level'].map(degree_map)
    all_data['cur_degree_id_num'] = all_data['cur_degree_id'].map(degree_map)
    all_data['same_edu'] = (all_data['min_edu_level'] == all_data['cur_degree_id']).astype(int)
    all_data['gt_edu'] = (all_data['cur_degree_id_num'] >= all_data['min_edu_level_num']).astype(int)
    all_data['min_desire_salary_num'] = (all_data['min_desire_salary'] <= all_data['min_salary']).astype(int)
    all_data['min_cur_salary_num'] = (all_data['min_cur_salary'] <= all_data['min_salary']).astype(int)

    all_data['max_desire_salary_num'] = (all_data['max_desire_salary'] <= all_data['max_salary']).astype(int)
    all_data['max_cur_salary_num'] = (all_data['max_cur_salary'] <= all_data['max_salary']).astype(int)
    all_data['same_desire_industry'] = all_data.apply(cur_industry_in_desire, axis=1).astype(int)
    all_data['same_jd_sub'] = all_data.apply(desire_in_jd, axis=1).astype(int)

    all_data['start_month'] = all_data['start_date'].dt.month
    all_data['start_day'] = all_data['start_date'].dt.day
    all_data['end_month'] = all_data['start_date'].dt.month
    all_data['end_day'] = all_data['start_date'].dt.day
    all_data['jd_days'] = (all_data['end_date'] - all_data['start_date']).dt.days

    all_data['user_work_year'] = 2019 - all_data['start_work_date'].replace('-', np.nan).astype(float)
    all_data['gt_min_year'] = (all_data['user_work_year'] > all_data['min_work_year']).astype(int)
    all_data['gt_max_year'] = (all_data['user_work_year'] > all_data['max_work_year']).astype(int)
    all_data['len_experience'] = all_data['experience'].apply(
        lambda x: len(x.split('|')) if isinstance(x, str) else np.nan)
    all_data['desire_jd_industry_id_len'] = all_data['desire_jd_industry_id'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    all_data['desire_jd_type_id_len'] = all_data['desire_jd_type_id'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)
    all_data['eff_exp_cnt'] = all_data.apply(jieba_cnt, axis=1)
    all_data['eff_exp_ratio'] = all_data['eff_exp_cnt'] / all_data['len_experience']
    all_data.drop(['cur_degree_id_num', 'cur_degree_id', 'desire_jd_city_id', 'min_years',
                   'start_work_date', 'start_date', 'end_date', 'key', 'min_edu_level'], axis=1, inplace=True)

    # 城市统计
    all_data['user_jd_city_nunique'] = all_data.groupby('user_id')['city'].transform('nunique').values
    all_data['jd_user_city_nunique'] = all_data.groupby('jd_no')['live_city_id'].transform('nunique').values

    all_data['jd_title_nunique'] = all_data.groupby('user_id')['jd_title'].transform('nunique').values
    all_data['jd_sub_type_nunique'] = all_data.groupby('user_id')['jd_sub_type'].transform('nunique').values

    all_data['user_desire_jd_industry_id_nunique'] = all_data.groupby('jd_no')['desire_jd_industry_id'].transform(
        'nunique').values
    all_data['user_desire_jd_type_id_nunique'] = all_data.groupby('jd_no')['desire_jd_type_id'].transform(
        'nunique').values

    # 薪资
    all_data['user_jd_min_salary_min'] = all_data.groupby('user_id')['min_salary'].transform('min').values
    all_data['user_jd_min_salary_max'] = all_data.groupby('user_id')['min_salary'].transform('max').values
    all_data['user_jd_min_salary_mean'] = all_data.groupby('user_id')['min_salary'].transform('mean').values
    all_data['user_jd_min_salary_std'] = all_data.groupby('user_id')['min_salary'].transform('std').values

    all_data['user_jd_max_salary_min'] = all_data.groupby('user_id')['max_salary'].transform('min').values
    all_data['user_jd_max_salary_max'] = all_data.groupby('user_id')['max_salary'].transform('max').values
    all_data['user_jd_max_salary_mean'] = all_data.groupby('user_id')['max_salary'].transform('mean').values
    all_data['user_jd_max_salary_std'] = all_data.groupby('user_id')['max_salary'].transform('std').values

    all_data['jd_user_desire_min_salary_min'] = all_data.groupby('jd_no')['min_desire_salary'].transform('min').values
    all_data['jd_user_desire_min_salary_max'] = all_data.groupby('jd_no')['min_desire_salary'].transform('max').values
    all_data['jd_user_desire_min_salary_mean'] = all_data.groupby('jd_no')['min_desire_salary'].transform('mean').values
    all_data['jd_user_desire_min_salary_std'] = all_data.groupby('jd_no')['min_desire_salary'].transform('std').values

    all_data['jd_user_desire_max_salary_min'] = all_data.groupby('jd_no')['max_desire_salary'].transform('min').values
    all_data['jd_user_desire_max_salary_max'] = all_data.groupby('jd_no')['max_desire_salary'].transform('max').values
    all_data['jd_user_desire_max_salary_mean'] = all_data.groupby('jd_no')['max_desire_salary'].transform('mean').values
    all_data['jd_user_desire_max_salary_std'] = all_data.groupby('jd_no')['max_desire_salary'].transform('std').values

    all_data['jd_days_min'] = all_data.groupby('user_id')['jd_days'].transform('min').values
    all_data['jd_days_max'] = all_data.groupby('user_id')['jd_days'].transform('max').values
    all_data['jd_days_mean'] = all_data.groupby('user_id')['jd_days'].transform('mean').values
    all_data['jd_days_std'] = all_data.groupby('user_id')['jd_days'].transform('std').values
    all_data['jd_days_skew'] = all_data.groupby('user_id')['jd_days'].transform('skew').values

    all_data['age_min'] = all_data.groupby('jd_no')['birthday'].transform('min').values
    all_data['age_max'] = all_data.groupby('jd_no')['birthday'].transform('max').values
    all_data['age_mean'] = all_data.groupby('jd_no')['birthday'].transform('mean').values
    all_data['age_std'] = all_data.groupby('jd_no')['birthday'].transform('std').values
    all_data['age_skew'] = all_data.groupby('jd_no')['birthday'].transform('skew').values

    for j in ['jd_title', 'jd_sub_type']:
        le = LabelEncoder()
        all_data[j].fillna('nan', inplace=True)
        all_data[f'{j}_map_num'] = le.fit_transform(all_data[j])

    all_data['experience'] = all_data['experience'].apply(lambda x: ' '.join(x.split('|') if
                                                                             isinstance(x, str) else 'nan'))
    exp_gp = all_data.groupby('jd_no')['experience'].agg(lambda x: ' '.join(x.to_list())).reset_index()
    exp_gp = get_tfidf(exp_gp, 'experience', 'jd_no')
    all_data = all_data.merge(exp_gp, on='jd_no', how='left')
    
    # 使用改进的胜任力维度特征提取
    print('添加增强的胜任力维度特征...')
    all_data = add_competency_dimension_features(all_data, stop_words)
    
    # 定义胜任力特征列表
    competency_features = [col for col in all_data.columns if col.startswith('comp_')]
    print(f'生成了{len(competency_features)}个胜任力特征')

    # 定义模型使用的所有特征
    use_feats = [c for c in all_data.columns if c not in ['user_id', 'jd_no', 'delivered', 'satisfied'] +
                 ['desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id', 'cur_jd_type', 'experience',
                 'jd_title', 'jd_sub_type', 'job_description\n']]
    
    # 确保胜任力特征包含在模型特征中
    print('添加胜任力特征到模型特征列表...')
    for feat in competency_features:
        if feat not in use_feats:
            use_feats.append(feat)
    
    # 定义胜任力维度映射
    competency_dimensions = {
        'knowledge_skills': [col for col in competency_features if any(x in col for x in 
                                                       ['skill', 'education', 'work_year', 'edu_exp', 'capability'])],
        'social_role': [col for col in competency_features if any(x in col for x in 
                                                    ['industry', 'job_type', 'career_path'])],
        'self_concept': [col for col in competency_features if any(x in col for x in 
                                                     ['salary', 'location'])],
        'traits': [col for col in competency_features if any(x in col for x in 
                                                ['text_similarity', 'skill_text', 'trait'])],
        'motive': [col for col in competency_features if any(x in col for x in 
                                                ['browse', 'exploration', 'focus', 'motive'])]
    }
    
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
    # 修复：确保从原始数据中获取delivered列
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
    
    # 测试不同的胜任力权重影响 - 从极小值开始，增加精度
    balance_factors = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    results = []
    
    for factor in balance_factors:
        try:
            # 计算自适应权重
            sat_weight, dev_weight, dim_weights, _, _ = calculate_adaptive_weights(
                sat_feat_imp, dev_feat_imp, competency_dimensions, factor, 0.7, dim_boost)
            
            # 合并预测数据并应用权重
            # 1. 合并胜任力维度特征到训练集
            train_pred_sat_merged = train_pred_sat.merge(
                all_data[all_data['satisfied'] != -1][['user_id', 'jd_no', 'delivered'] + 
                                                    [f'comp_dim_{dim}' for dim in competency_dimensions.keys()] +
                                                    ['competency_overall_score']],
                on=['user_id', 'jd_no'], 
                how='left'
            )
            
            # 2. 合并投递预测结果
            train_pred_sat_merged = train_pred_sat_merged.merge(
                train_pred_dev[['user_id', 'jd_no', 'delivered_pred']],
                on=['user_id', 'jd_no'],
                how='left'
            )
            
            # 3. 应用权重计算综合预测分数
            train_pred_sat_merged['temp_pred'] = train_pred_sat_merged['satisfied_pred'] * sat_weight + \
                                              train_pred_sat_merged['delivered_pred'] * dev_weight
            
            # 4. 添加各胜任力维度的加权分数
            for dim in competency_dimensions.keys():
                dim_col = f'comp_dim_{dim}'
                if dim_col in train_pred_sat_merged.columns:
                    train_pred_sat_merged['temp_pred'] += train_pred_sat_merged[dim_col] * dim_weights[dim]
            
            # 5. 评估结果
            if check_columns_exist(train_pred_sat_merged, ['delivered', 'temp_pred', 'satisfied']):
                temp_dev_map = offline_eval_map(train_pred_sat_merged, 'delivered', 'temp_pred')
                temp_sat_map = offline_eval_map(train_pred_sat_merged, 'satisfied', 'temp_pred')
                temp_final = 0.7 * temp_sat_map + 0.3 * temp_dev_map
                results.append((factor, temp_sat_map, temp_dev_map, temp_final))
            else:
                print(f"警告: 缺少评估必要的列，跳过权重因子 {factor} 的评估")
        except Exception as e:
            print(f"处理权重因子 {factor} 时出错: {e}")
    
    # 显示不同胜任力权重系数的结果
    if results:
        print("\n不同胜任力权重系数的结果:")
        print("胜任力权重系数 | 满意度MAP | 投递MAP | 最终得分")
        print("-" * 50)
        for factor, sat_map, dev_map, final in results:
            print(f"{factor:.3f} | {sat_map:.4f} | {dev_map:.4f} | {final:.4f}")
        
        # 选择最佳权重系数
        best_result = max(results, key=lambda x: x[3])
        print(f"\n最佳胜任力权重系数: {best_result[0]:.3f}, 满意度MAP: {best_result[1]:.4f}, " +
            f"投递MAP: {best_result[2]:.4f}, 最终得分: {best_result[3]:.4f}")
        
        # 使用最佳权重系数应用到测试集
        best_factor = best_result[0]
        best_sat_weight, best_dev_weight, best_dim_weights, _, _ = calculate_adaptive_weights(
            sat_feat_imp, dev_feat_imp, competency_dimensions, best_factor, 0.7, dim_boost)
        
        # 合并胜任力维度特征到测试集预测结果
        sub_sat_comp = sub_sat.merge(
            all_data[all_data['satisfied'] == -1][['user_id', 'jd_no'] + 
                                                [f'comp_dim_{dim}' for dim in competency_dimensions.keys()] +
                                                ['competency_overall_score']],
            on=['user_id', 'jd_no'],
            how='left'
        )
        
        sub_dev_comp = sub_dev.merge(
            all_data[all_data['delivered'] == -1][['user_id', 'jd_no'] + 
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
                sub_sat_comp['best_merge_prob'] += sub_sat_comp[dim_col] * best_dim_weights[dim]
        
        # 输出最终结果 - 使用最佳权重
        print("\n生成最终提交结果...")
        sub_df = pd.DataFrame(columns=['user_id', 'jd_no', 'best_merge_prob'])
        for i in sub_sat_comp['user_id'].unique():
            # 排序确保高匹配度职位优先推荐
            tmp_sub = sub_sat_comp[(sub_sat_comp['user_id'] == i) &
                            (sub_sat_comp['jd_no'].isin(train_jd['jd_no']))].sort_values('best_merge_prob', ascending=False)[
                            ['user_id', 'jd_no', 'best_merge_prob']]
            sub_df = pd.concat([sub_df, tmp_sub], ignore_index=True)
            sub_df = pd.concat([sub_df, sub_sat_comp[(sub_sat_comp['user_id'] == i) & (~sub_sat_comp['jd_no'].isin(train_jd['jd_no']))][
                    ['user_id', 'jd_no', 'best_merge_prob']]], ignore_index=True)
        
        # 保存最终结果
        sub_df[['user_id', 'jd_no']].to_csv('sub_with_enhanced_competency_optimized.csv', index=False)
        
        # 保存特征重要性分析结果和详细调试信息
        sat_feat_imp[['Feature', 'avg_imp']].to_csv('feat_imp_satisfied_full.csv', index=False, encoding='utf8')
        dev_feat_imp[['Feature', 'avg_imp']].to_csv('feat_imp_delivered_full.csv', index=False, encoding='utf8')
        
        # 保存最佳模型配置信息
        config = {
            'best_factor': best_factor,
            'best_sat_weight': best_sat_weight,
            'best_dev_weight': best_dev_weight,
            'dim_boost': str(dim_boost),
            'final_score': best_result[3]
        }
        
        # 将配置转为DataFrame便于保存
        config_df = pd.DataFrame([config])
        config_df.to_csv('best_model_config.csv', index=False)
        
        print("\n所有处理完成!")
        print(f"最终得分: {best_result[3]:.4f}")
    else:
        print("警告: 未能获得有效的评估结果，无法确定最佳权重")