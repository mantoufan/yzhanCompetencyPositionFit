import pandas as pd
import numpy as np
import re
import json
import networkx as nx
import requests
import os
from difflib import SequenceMatcher
import hashlib
import pickle
import atexit

class CacheManager:
    """缓存管理器，用于存储和检索实体抽取和语义模型的结果"""
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"创建缓存目录: {cache_dir}")
        
        # 创建子目录
        self.resume_cache_dir = os.path.join(cache_dir, 'resumes')
        self.jd_cache_dir = os.path.join(cache_dir, 'jobs')
        self.entity_cache_dir = os.path.join(cache_dir, 'entities')  # 新增实体缓存目录
        
        for directory in [self.resume_cache_dir, self.jd_cache_dir, self.entity_cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建缓存子目录: {directory}")
    
    def _generate_cache_key(self, text):
        """生成缓存键，使用文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_cached_model(self, text, entity_type):
        """获取缓存的模型，如果不存在则返回None"""
        cache_key = self._generate_cache_key(text)
        cache_dir = self.resume_cache_dir if entity_type == "resume" else self.jd_cache_dir
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_model = pickle.load(f)
                print(f"从缓存加载{entity_type}模型: {cache_key[:8]}...")
                return cached_model
            except Exception as e:
                print(f"读取缓存文件时出错: {e}")
                return None
        return None
    
    def save_model_to_cache(self, text, entity_type, model):
        """保存模型到缓存"""
        cache_key = self._generate_cache_key(text)
        cache_dir = self.resume_cache_dir if entity_type == "resume" else self.jd_cache_dir
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"保存{entity_type}模型到缓存: {cache_key[:8]}...")
            return True
        except Exception as e:
            print(f"保存缓存文件时出错: {e}")
            return False
            
    def get_cached_entity_registry(self):
        """获取缓存的实体注册表，如果不存在则返回None"""
        cache_file = os.path.join(self.entity_cache_dir, "entity_registry.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_registry = pickle.load(f)
                print(f"从缓存加载实体注册表...")
                return cached_registry
            except Exception as e:
                print(f"读取实体注册表缓存文件时出错: {e}")
                return None
        return None
    
    def save_entity_registry_to_cache(self, registry):
        """保存实体注册表到缓存"""
        cache_file = os.path.join(self.entity_cache_dir, "entity_registry.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(registry, f)
            print(f"保存实体注册表到缓存...")
            return True
        except Exception as e:
            print(f"保存实体注册表缓存文件时出错: {e}")
            return False

# 初始化缓存管理器
cache_manager = CacheManager()

# =============================================
# 第一部分：实体注册表（支持动态分类）
# =============================================
class EntityRegistry:
    """实体注册表类，用于管理和消歧实体，支持动态添加新分类"""
    def __init__(self):
        # 初始基本类别
        self.entities = {
            "技能": {},              # 原skills
            "工作经验": {},          # 原experience
            "教育背景": {},          # 原education
            "行业领域": {},          # 原industry
            "职位要求": {},          # 原requirements
            "工作职责": {}           # 原responsibilities
        }
        self.category_descriptions = {
            "技能": "专业技能、技术能力、工具使用等",
            "工作经验": "工作经历、项目经验等",
            "教育背景": "学历、专业背景、培训经历等",
            "行业领域": "行业领域、专业方向等",
            "职位要求": "职位要求、招聘条件等",
            "工作职责": "工作职责、岗位任务等"
        }
        self.canonical_forms = {}   # 规范形式的映射表
    
    def add_category(self, category_name, category_description=""):
        """动态添加新的实体类别"""
        if category_name.lower() not in [c.lower() for c in self.entities.keys()]:
            # 检查是否有类似的类别名称
            for existing_category in self.entities.keys():
                if SequenceMatcher(None, category_name.lower(), existing_category.lower()).ratio() > 0.8:
                    print(f"新类别 '{category_name}' 与已有类别 '{existing_category}' 相似，使用已有类别")
                    return existing_category
            
            # 验证新类别名称的有效性（名称不应太长，不应包含特殊字符等）
            if len(category_name) > 20 or any(char in category_name for char in "!@#$%^&*()+=[]{}|\\;:'\",.<>/?"):
                print(f"忽略无效的类别名称: {category_name}")
                return None
                
            self.entities[category_name] = {}
            self.category_descriptions[category_name] = category_description
            print(f"新增实体类别: {category_name} - {category_description}")
            return category_name
        return None
    
    def add_entity(self, category, entity_name, min_entity_length=2, max_entity_length=50):
        """添加实体到清单中，如有相似实体则返回规范形式"""
         # 验证实体名称的有效性
        if not isinstance(entity_name, str):
            return None
        
        # 检查实体长度
        if len(entity_name) < min_entity_length or len(entity_name) > max_entity_length:
            return None
            
        # 如果实体已经有规范形式，直接返回
        if entity_name in self.canonical_forms:
            return self.canonical_forms[entity_name]
        
        # 检查是否有相似实体
        best_match = None
        best_similarity = 0.0
        threshold = 0.85  # 相似度阈值
        
        if category in self.entities:
            for existing_entity in self.entities[category].keys():
                similarity = SequenceMatcher(None, entity_name.lower(), existing_entity.lower()).ratio()
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_entity
        
        # 如果有相似实体，使用已有的规范形式
        if best_match:
            self.canonical_forms[entity_name] = best_match
            return best_match
        
        # 如果没有相似实体，添加为新实体
        if category in self.entities:
            self.entities[category][entity_name] = 1
            self.canonical_forms[entity_name] = entity_name
        
        return entity_name
    
    def normalize_entities(self, category, entity_list):
        """规范化实体列表"""
        normalized = []
        for entity in entity_list:
            normalized_entity = self.add_entity(category, entity)
            if normalized_entity:  # 只添加有效的实体
                normalized.append(normalized_entity)
        return normalized

    def get_all_entities(self, category=None):
        """获取所有实体或特定类别的实体"""
        if category:
            return list(self.entities.get(category, {}).keys())
        else:
            all_entities = []
            for category in self.entities:
                all_entities.extend(list(self.entities[category].keys()))
            return all_entities
            
    def get_all_categories(self):
        """获取所有实体类别及其描述"""
        return {cat: desc for cat, desc in self.category_descriptions.items()}
    
    def clean_empty_categories(self, min_entities=1):
        """清理没有足够实体的类别，保留核心类别"""
        # 核心类别，不会被删除
        core_categories = {"技能", "工作经验", "教育背景", "行业领域", "职位要求", "工作职责"}
        
        categories_to_remove = []
        for category, entities in self.entities.items():
            if category not in core_categories and len(entities) < min_entities:
                categories_to_remove.append(category)
        
        for category in categories_to_remove:
            del self.entities[category]
            if category in self.category_descriptions:
                del self.category_descriptions[category]
            print(f"清理空类别: {category}")
        
        return len(categories_to_remove)

# 初始化全局实体清单
entity_registry = EntityRegistry()

def call_gpt_api(prompt):
    """调用gpt-4o-mini模型API进行推理"""
    api_url = "https://api.gueai.com/v1/chat/completions"
    api_key = "sk-Qbu2xS4mZ5iFLfMa1UM162EE2epH6IbNN0rRrDvv2RJ4589U"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "你是一个专业的实体和关系抽取助手，擅长从文本中提取关键信息并以JSON格式返回。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return response_text
        else:
            print(f"API调用失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None
    except Exception as e:
        print(f"调用模型时发生错误: {e}")
        return None

# =============================================
# 第二部分：数据加载与预处理
# =============================================
def load_data(data_dir='./data'):
    """加载三个表格数据，处理可能的CSV格式问题和分隔符问题"""
    
    # 确保目录路径正确
    if not os.path.exists(data_dir):
        print(f"警告: 目录 {data_dir} 不存在，尝试直接从当前目录加载文件")
        data_dir = '.'
    
    user_file = os.path.join(data_dir, 'table1_user.csv')
    jd_file = os.path.join(data_dir, 'table2_jd.csv')
    action_file = os.path.join(data_dir, 'table3_action.csv')
    
    # 检查文件是否存在
    for file_path, file_type in [(user_file, '用户'), (jd_file, '职位'), (action_file, '行为')]:
        if not os.path.exists(file_path):
            print(f"警告: {file_type}数据文件 {file_path} 不存在")
    
    try:
        # 尝试使用更灵活的参数读取用户CSV
        try:
            user_df = pd.read_csv(user_file, encoding='utf-8', low_memory=False)
        except Exception as e:
            print(f"标准方式读取用户数据失败，尝试更宽松的解析: {e}")
            user_df = pd.read_csv(user_file, encoding='utf-8', low_memory=False, 
                               engine='python', on_bad_lines='skip')
        
        print(f"成功加载用户数据：{len(user_df)}条记录")
        
        # 对于职位数据，尝试使用不同的分隔符
        print("尝试加载职位数据...")
        
        # 首先检查文件的前几行，判断可能的分隔符
        with open(jd_file, 'r', encoding='utf-8', errors='replace') as f:
            sample_lines = [next(f) for _ in range(5) if f]
        
        # 检查可能的分隔符
        comma_count = sample_lines[0].count(',')
        tab_count = sample_lines[0].count('\t')
        semicolon_count = sample_lines[0].count(';')
        
        print(f"检测到可能的分隔符: 逗号({comma_count}), 制表符({tab_count}), 分号({semicolon_count})")
        
        # 根据检测结果选择分隔符
        separator = ','  # 默认分隔符
        if tab_count > comma_count and tab_count > semicolon_count:
            separator = '\t'
            print(f"使用制表符(\\t)作为分隔符加载职位数据")
        elif semicolon_count > comma_count and semicolon_count > tab_count:
            separator = ';'
            print(f"使用分号(;)作为分隔符加载职位数据")
        else:
            print(f"使用逗号(,)作为分隔符加载职位数据")
        
        # 尝试用检测到的分隔符加载数据
        try:
            jd_df = pd.read_csv(jd_file, sep=separator, encoding='utf-8', 
                             engine='python', on_bad_lines='skip')
            
            # 检查列名情况
            if len(jd_df.columns) == 1 and '\t' in jd_df.columns[0]:
                print("检测到列名包含制表符，进行拆分处理...")
                
                # 列名可能被错误地合并了，需要拆分
                column_names = jd_df.columns[0].split('\t')
                
                # 如果数据也被合并了，需要拆分每一行
                if all(isinstance(val, str) and '\t' in val for val in jd_df.iloc[:5, 0] if isinstance(val, str)):
                    print("数据行也需要拆分，进行处理...")
                    
                    # 读取原始文件
                    with open(jd_file, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                    
                    # 拆分每一行
                    data = []
                    for line in lines[1:]:  # 跳过头部
                        row = line.strip().split('\t')
                        data.append(row)
                    
                    # 创建新的DataFrame
                    jd_df = pd.DataFrame(data, columns=column_names)
                else:
                    # 只有列名需要拆分
                    jd_df.columns = column_names
                    
            # 检查job_description列，可能包含多余的引号
            if '"job_description' in jd_df.columns:
                # 移除列名中的引号
                jd_df.rename(columns={'"job_description': 'job_description'}, inplace=True)
            
            # 检查最后一列，可能包含引号
            last_col = jd_df.columns[-1]
            if last_col.startswith('"') or last_col.endswith('"'):
                # 移除列名中的引号
                clean_col = last_col.strip('"')
                jd_df.rename(columns={last_col: clean_col}, inplace=True)
                
        except Exception as e:
            print(f"使用分隔符'{separator}'加载失败: {e}")
            print("尝试手动解析职位数据...")
            
            # 手动读取并解析文件
            with open(jd_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # 假设第一行是头部，使用制表符分割
            header = lines[0].strip().split('\t')
            header = [h.strip('"') for h in header]  # 移除可能的引号
            
            # 准备数据
            data = []
            for line in lines[1:]:
                row = line.strip().split('\t')
                # 确保每行的字段数与头部一致
                if len(row) > len(header):
                    # 合并多余字段
                    row = row[:len(header)-1] + ['\t'.join(row[len(header)-1:])]
                elif len(row) < len(header):
                    # 用空值填充不足字段
                    row = row + [''] * (len(header) - len(row))
                data.append(row)
            
            # 创建DataFrame
            jd_df = pd.DataFrame(data, columns=header)
        
        print(f"成功加载职位数据：{len(jd_df)}条记录")
        print(f"职位数据列名: {list(jd_df.columns)}")
        
        # 对于行为数据采用相同的策略
        try:
            action_df = pd.read_csv(action_file, encoding='utf-8')
        except Exception as e:
            print(f"标准方式读取行为数据失败，尝试更宽松的解析: {e}")
            action_df = pd.read_csv(action_file, encoding='utf-8', 
                                 engine='python', on_bad_lines='skip')
        
        print(f"成功加载行为数据：{len(action_df)}条记录")
        
        # 确保jd_df中有job_description列和jd_no列
        # 检查job_description列
        desc_found = False
        for col in jd_df.columns:
            if 'description' in col.lower() or 'desc' in col.lower():
                if col != 'job_description':
                    print(f"将列'{col}'重命名为'job_description'")
                    jd_df.rename(columns={col: 'job_description'}, inplace=True)
                desc_found = True
                break
        
        if not desc_found and len(jd_df.columns) > 0:
            # 尝试使用最后一列作为job_description
            last_col = jd_df.columns[-1]
            print(f"未找到description列，将最后一列'{last_col}'设为'job_description'")
            jd_df.rename(columns={last_col: 'job_description'}, inplace=True)
        
        # 确保有jd_no列
        id_found = False
        for col in jd_df.columns:
            if 'id' in col.lower() or 'no' in col.lower():
                if col != 'jd_no':
                    print(f"将列'{col}'重命名为'jd_no'")
                    jd_df.rename(columns={col: 'jd_no'}, inplace=True)
                id_found = True
                break
        
        if not id_found and len(jd_df.columns) > 0:
            # 使用第一列作为jd_no
            first_col = jd_df.columns[0]
            print(f"未找到id列，将第一列'{first_col}'设为'jd_no'")
            jd_df.rename(columns={first_col: 'jd_no'}, inplace=True)
        
        # 打印最终列名
        print(f"最终用户数据列名: {list(user_df.columns)}")
        print(f"最终职位数据列名: {list(jd_df.columns)}")
        print(f"最终行为数据列名: {list(action_df.columns)}")
        
        return user_df, jd_df, action_df
        
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        # 返回空的DataFrame以避免程序崩溃
        empty_df = pd.DataFrame()
        return empty_df.copy(), empty_df.copy(), empty_df.copy()

def manual_parse_csv(file_path):
    """手动解析可能有问题的CSV文件"""
    print(f"开始手动解析文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 假设第一行是头部
        header = lines[0].strip().split(',')
        print(f"检测到表头: {header}")
        
        # 准备数据列表
        data = []
        for i, line in enumerate(lines[1:], 2):
            try:
                # 尝试解析每一行
                row = line.strip().split(',')
                
                # 如果字段数量与表头不匹配，处理这种情况
                if len(row) != len(header):
                    print(f"第{i}行字段数量({len(row)})与表头({len(header)})不匹配")
                    
                    if len(row) > len(header):
                        # 合并多余字段
                        extra_fields = ','.join(row[len(header)-1:])
                        row = row[:len(header)-1] + [extra_fields]
                    else:
                        # 如果字段不足，用空值填充
                        row.extend([''] * (len(header) - len(row)))
                
                data.append(row)
            except Exception as line_error:
                print(f"解析第{i}行时出错: {line_error}")
                print(f"问题行内容: {line}")
                # 继续处理下一行
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=header)
        print(f"手动解析完成，成功解析{len(df)}行数据")
        return df
        
    except Exception as e:
        print(f"手动解析失败: {e}")
        # 返回空DataFrame
        return pd.DataFrame()

def preprocess_text(text):
    """文本预处理：清洗文本"""
    if pd.isna(text):
        return ""
    # 清洗文本
    text = re.sub(r'\n|\\n', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def initialize_entity_registry(user_df, jd_df):
    """初始化实体清单，从数据中预先提取常见实体"""
   # 从用户简历中提取关键词
    for _, user in user_df.head(10).iterrows():  # 只处理前10条用户数据用于初始化
        if 'experience' in user:
            experience_text = str(user['experience'])
            if len(experience_text) > 10:  # 只处理有意义的文本
                # 简单分词并添加到实体清单
                words = experience_text.split('|')
                for word in words:
                    if len(word) > 1:
                        if len(word) <= 10:  # 只添加合理长度的实体
                            entity_registry.add_entity("技能", word)
                        elif len(word) > 10 and "|" in word:
                            # 处理可能的多个实体
                            sub_words = word.split("|")
                            for sub_word in sub_words:
                                if len(sub_word) > 1 and len(sub_word) <= 10:
                                    entity_registry.add_entity("技能", sub_word)
    
    # 从职位描述中提取关键词
    for _, job in jd_df.head(5).iterrows():  # 只处理前5条职位数据用于初始化
        if 'job_description' in job:
            job_text = str(job['job_description'])
            if len(job_text) > 50:  # 只处理有意义的文本
                # 提取常见要求关键词
                common_requirements = ['本科', '大专', '经验', '团队', '沟通', '解决问题', 
                                      '责任心', '学习能力', 'office', '项目管理']
                for req in common_requirements:
                    if req in job_text:
                        if '经验' in req:
                            entity_registry.add_entity("工作经验", req)
                        elif req in ['本科', '大专']:
                            entity_registry.add_entity("教育背景", req)
                        else:
                            entity_registry.add_entity("技能", req)
    
    # 添加一些基本的教育实体
    education_entities = ['本科', '大专', '硕士', '博士', '高中', '中专', 
                         '计算机科学', '软件工程', '工商管理', '市场营销']
    for edu in education_entities:
        entity_registry.add_entity("教育背景", edu)
    
    # 添加一些基本的行业实体
    industry_entities = ['互联网', '金融', '教育', '医疗', '房地产', '制造业', 
                        '零售', '物流', '咨询', '广告', '建筑']
    for ind in industry_entities:
        entity_registry.add_entity("行业领域", ind)
    
    # 添加一些基本的职位要求和职责
    requirement_entities = ['团队合作', '沟通能力', '解决问题能力', '学习能力', 
                           '责任心', '执行力', '创新能力', '领导力']
    for req in requirement_entities:
        entity_registry.add_entity("职位要求", req)
    
    responsibility_entities = ['项目管理', '团队管理', '客户沟通', '数据分析', 
                              '产品开发', '市场推广', '销售']
    for resp in responsibility_entities:
        entity_registry.add_entity("工作职责", resp)

# =============================================
# 第三部分：动态发现新实体类别
# =============================================
def discover_new_categories(text, entity_type):
    """分析文本，发现可能的新实体类别，增加验证逻辑"""
    existing_categories = list(entity_registry.entities.keys())
    existing_categories_json = json.dumps({
        category: entity_registry.category_descriptions.get(category, "")
        for category in existing_categories
    }, ensure_ascii=False)
    
    prompt = f"""
    请分析以下{("简历" if entity_type == "resume" else "职位描述")}文本，识别出可能的新实体类别。
    
    文本内容："{text}"
    
    系统当前已有的实体类别：
    {existing_categories_json}
    
    请找出文本中可能存在但当前类别中未包含的新实体类别。每个新类别应包含名称和简短描述。
    
    注意:
    1. 只返回真正有必要的、明确的新类别，避免创建过于细分或重叠的类别
    2. 确保新类别至少有3个以上的实体实例在文本中存在
    3. 新类别应该是招聘和人才匹配领域普遍使用的术语
    4. 类别名称应简短明确，不超过5个汉字
    
    请以JSON格式返回结果：
    {{
        "new_categories": [
            {{"name": "类别名称1", "description": "类别描述1", "examples": ["示例实体1", "示例实体2", "示例实体3"]}},
            {{"name": "类别名称2", "description": "类别描述2", "examples": ["示例实体1", "示例实体2", "示例实体3"]}},
            ...
        ]
    }}
    
    如果没有发现新类别，请返回空列表。请确保新类别是有意义的，且与招聘和人才匹配相关。
    """
    
    # 调用gpt-4o-mini模型进行分析
    response_text = call_gpt_api(prompt)
    
    if response_text:
        # 提取JSON部分
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            categories_json = json_match.group(1)
            try:
                categories_data = json.loads(categories_json)
                
                # 添加新类别，并验证是否有示例实体
                added_categories = []
                for new_category in categories_data.get("new_categories", []):
                    category_name = new_category.get("name", "").strip()
                    category_desc = new_category.get("description", "").strip()
                    examples = new_category.get("examples", [])
                    
                    # 验证类别的有效性：至少有3个示例实体
                    if category_name and len(examples) >= 3:
                        added_category = entity_registry.add_category(category_name, category_desc)
                        if added_category:
                            added_categories.append(added_category)
                            
                            # 添加示例实体到类别中
                            for example in examples[:5]:  # 限制最多添加5个示例
                                entity_registry.add_entity(added_category, example)
                                
                return added_categories
            except json.JSONDecodeError as e:
                print(f"解析新类别JSON时出错: {e}")
                return []
    return []

# =============================================
# 第四部分：实体和关系抽取
# =============================================
def extract_entities_with_llm(text, entity_type):
    """使用大语言模型进行实体抽取，支持动态类别"""
    # 首先尝试发现新的实体类别
    new_categories = discover_new_categories(text, entity_type)
    if new_categories:
        print(f"发现并添加了{len(new_categories)}个新实体类别: {', '.join(new_categories)}")
    
    # 获取当前所有类别及描述
    all_categories = entity_registry.get_all_categories()
    
    # 构建类别结构JSON
    categories_json = json.dumps({
        cat: {"description": desc, "examples": entity_registry.get_all_entities(cat)[:5]}
        for cat, desc in all_categories.items()
    }, ensure_ascii=False)
    
    # 构建向大模型的提示
    if entity_type == "resume":
        prompt = f"""
        请从以下简历文本中提取关键信息，并以JSON格式返回，按照以下类别进行分类。
        
        当前系统支持的实体类别：
        {categories_json}
        
        简历文本："{text}"
        
        请按照上述类别提取实体，并以JSON格式返回结果。对于每个类别，提取相关的实体列表。
        如果某个类别没有找到相关实体，返回空列表。
        
        提取原则：
        1. 每个实体应该是具体、明确的技能、经验或资质，而不是模糊的描述
        2. 实体表述应该简洁，通常不超过10个字
        3. 对于不确定的实体，不要强行归类，宁缺毋滥
        4. 专注于核心类别（技能、工作经验、教育背景、行业领域等）
        5. 尽量使用规范术语，避免非标准表达
        
        尽量使用已有的规范实体表述，对于新发现的实体，使用最规范、最简洁的表达形式。
        
        请以JSON格式返回，格式如下：
        {{
            "类别1": ["实体1", "实体2", ...],
            "类别2": ["实体1", "实体2", ...],
            ...
        }}
        
        仅返回JSON，不要有其他解释。
        """
    else:  # job description
        prompt = f"""
        请从以下职位描述文本中提取关键信息，并以JSON格式返回，按照以下类别进行分类。
        
        当前系统支持的实体类别：
        {categories_json}
        
        职位描述文本："{text}"
        
        请按照上述类别提取实体，并以JSON格式返回结果。对于每个类别，提取相关的实体列表。
        如果某个类别没有找到相关实体，返回空列表。
        
        提取原则：
        1. 每个实体应该是具体、明确的要求、职责或资质，而不是模糊的描述
        2. 实体表述应该简洁，通常不超过10个字
        3. 对于不确定的实体，不要强行归类，宁缺毋滥
        4. 专注于核心类别（职位要求、工作职责、技能、工作经验等）
        5. 尽量使用规范术语，避免非标准表达
        
        特别注意，对于职位描述，通常包含"职位要求"和"工作职责"两个核心类别。
        
        尽量使用已有的规范实体表述，对于新发现的实体，使用最规范、最简洁的表达形式。
        
        请以JSON格式返回，格式如下：
        {{
            "类别1": ["实体1", "实体2", ...],
            "类别2": ["实体1", "实体2", ...],
            ...
        }}
        
        仅返回JSON，不要有其他解释。
        """
    
    # 调用gpt-4o-mini模型
    response_text = call_gpt_api(prompt)
    
    if response_text:
        # 提取JSON部分
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            entities_json = json_match.group(1)
        else:
            entities_json = response_text
            
        # 解析JSON
        try:
            entities = json.loads(entities_json)
            
            # 通过实体清单进行规范化
            normalized_entities = {}
            for category, entity_list in entities.items():
                if entity_list:  # 只处理非空列表
                    normalized_entities[category] = entity_registry.normalize_entities(category, entity_list)
            
            return normalized_entities
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response_text}")
            
            # 返回默认结构
            return {"技能": [], "工作经验": [], "教育背景": [], "行业领域": []}
    else:
        return {"技能": [], "工作经验": [], "教育背景": [], "行业领域": []}

def extract_relations_with_llm(entities, entity_type):
    """使用大语言模型构建关系三元组，自动发现关系类型，并确保使用规范化实体"""
    # 将实体转换为字符串
    entities_str = json.dumps(entities, ensure_ascii=False)
    
    # 获取已有的实体列表用于消歧
    all_entities_list = entity_registry.get_all_entities()
    all_entities_json = json.dumps(all_entities_list, ensure_ascii=False)
    
    # 构建向大模型的提示
    if entity_type == "resume":
        prompt = f"""
        根据以下简历中提取的实体，构建关系三元组。 
        实体信息：{entities_str}
        
        请自动分析这些实体，发现适合的关系类型，并构建(主体, 关系, 客体)格式的三元组列表。
        
        要求：
        1. 主体可以是"person"或其他适合表示求职者的实体
        2. 关系应该准确描述主体与客体之间的联系，例如"具有技能"、"拥有经验"等
        3. 客体应该是实体信息中的具体内容
        4. 关系类型应根据实体的性质自动发现，不要局限于预设的关系类型
        5. 优先使用以下已存在的规范实体，确保实体表述的一致性：
        {all_entities_json}
        
        请以JSON格式返回三元组列表，格式如下：
        [
            ["主体", "关系", "客体"],
            ["主体", "关系", "客体"],
            ...
        ]
        """
    else:  # job description
        prompt = f"""
        根据以下职位描述中提取的实体，构建关系三元组。
        实体信息：{entities_str}
        
        请自动分析这些实体，发现适合的关系类型，并构建(主体, 关系, 客体)格式的三元组列表。
        
        要求：
        1. 主体可以是"job"或其他适合表示职位的实体
        2. 关系应该准确描述主体与客体之间的联系，例如"要求具备"、"职责包括"等
        3. 客体应该是实体信息中的具体内容
        4. 关系类型应根据实体的性质自动发现，不要局限于预设的关系类型
        5. 优先使用以下已存在的规范实体，确保实体表述的一致性：
        {all_entities_json}
        
        请以JSON格式返回三元组列表，格式如下：
        [
            ["主体", "关系", "客体"],
            ["主体", "关系", "客体"],
            ...
        ]
        """
    
    # 调用gpt-4o-mini模型
    response_text = call_gpt_api(prompt)
    
    if response_text:
        # 提取JSON部分
        json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
        if json_match:
            triples_json = json_match.group(1)
        else:
            triples_json = response_text
            
        # 解析JSON
        try:
            triples = json.loads(triples_json)
            
            # 规范化三元组中的实体
            normalized_triples = []
            for subj, rel, obj in triples:
                # 确保obj是字符串类型
                if isinstance(obj, list):
                    # 如果obj是列表，可以将其转换为字符串或只取第一个元素
                    obj = obj[0] if obj else ""  # 或使用 str(obj) 转为字符串
                
                # 尝试确定客体的类别
                obj_category = None
                for category in entity_registry.entities:
                    if obj in entity_registry.entities[category] or obj in entity_registry.canonical_forms:
                        obj_category = category
                        break
                
                # 如果找到类别，规范化客体
                if obj_category:
                    normalized_obj = entity_registry.add_entity(obj_category, obj)
                    normalized_triples.append([subj, rel, normalized_obj])
                else:
                    # 如果无法确定类别，使用原始实体
                    normalized_triples.append([subj, rel, obj])
            
            return normalized_triples
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            # 返回空列表
            return []
    else:
        print(f"API调用失败")
        return []
    
# =============================================
# 第五部分：胜任力模型树构建
# =============================================
def build_competency_tree_with_llm(entities, entity_type):
    """使用大语言模型构建胜任力模型的树表示，自动发现维度，并使用规范化实体"""
    # 将实体转换为字符串
    entities_str = json.dumps(entities, ensure_ascii=False)
    
    # 获取已有的实体列表用于消歧
    all_entities_list = entity_registry.get_all_entities()
    all_entities_json = json.dumps(all_entities_list, ensure_ascii=False)
    
    # 获取当前所有类别
    all_categories = list(entity_registry.entities.keys())
    categories_json = json.dumps(all_categories, ensure_ascii=False)
    
    # 构建向大模型的提示
    prompt = f"""
    请帮我根据以下{("简历" if entity_type == "resume" else "职位描述")}实体信息，构建一个胜任力模型的树结构。
    
    实体信息：{entities_str}
    
    实体类别：{categories_json}
    
    树结构应该满足以下要求：
    1. 根节点为"{entity_type}_root"
    2. 一级节点为自动发现的维度类别，请根据实体的特点和上述实体类别进行聚类，归纳出适合的维度（至少3个，不超过8个）
    3. 二级节点为具体的实体
    
    请分析实体信息，自动发现合适的维度类别，并将每个实体分配到最合适的维度下。
    请确保使用以下已存在的规范实体名称，保持实体表述的一致性：
    {all_entities_json}
    
    以JSON格式返回结果：
    {{
        "root": "{entity_type}_root",
        "dimensions": [
            {{
                "name": "维度名称1",
                "description": "该维度的简要描述",
                "entities": ["实体1", "实体2", ...]
            }},
            {{
                "name": "维度名称2",
                "description": "该维度的简要描述",
                "entities": ["实体3", "实体4", ...]
            }},
            ...
        ]
    }}
    
    请确保每个实体只分配到一个最合适的维度下，维度名称应简洁明了，描述应能清晰表达该维度的特点。
    """
    
    # 调用gpt-4o-mini模型
    response_text = call_gpt_api(prompt)
    
    if response_text:
        # 提取JSON部分
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            tree_json = json_match.group(1)
        else:
            tree_json = response_text
            
        try:
            # 解析JSON
            tree_structure = json.loads(tree_json)
            
            # 构建NetworkX图
            G = nx.DiGraph()
            
            # 添加根节点 - 修复这里，使用"root"键而不是"根节点"键
            # 兼容两种可能的键名
            if "root" in tree_structure:
                root = tree_structure["root"]
            elif "根节点" in tree_structure:
                root = tree_structure["根节点"]
            else:
                # 如果两个键都不存在，使用默认值
                root = f"{entity_type}_root"
                print(f"警告: 无法从API响应中找到根节点信息，使用默认值: {root}")
            
            G.add_node(root)
            
            # 添加维度节点和实体节点 - 同样修复这里，兼容两种可能的键名
            dimension_list = []  # 存储所有维度名称
            
            # 处理不同的维度节点键名
            if "dimensions" in tree_structure:
                dimensions_data = tree_structure["dimensions"]
            elif "维度节点" in tree_structure:
                dimensions_data = tree_structure["维度节点"]
            else:
                # 如果维度节点信息不存在，创建默认维度
                print(f"警告: 无法从API响应中找到维度节点信息，使用默认维度")
                dimensions_data = create_default_dimensions(entities)
            
            for dimension_info in dimensions_data:
                # 处理不同的维度名称键
                if "name" in dimension_info:
                    dimension = dimension_info["name"]
                elif "维度" in dimension_info:
                    dimension = dimension_info["维度"]
                else:
                    # 跳过缺少维度名称的项
                    print(f"警告: 跳过缺少维度名称的项: {dimension_info}")
                    continue
                
                # 处理不同的描述键
                if "description" in dimension_info:
                    dimension_desc = dimension_info["description"]
                elif "描述" in dimension_info:
                    dimension_desc = dimension_info["描述"]
                else:
                    dimension_desc = ""
                
                # 处理不同的实体列表键
                if "entities" in dimension_info:
                    entities_list = dimension_info["entities"]
                elif "实体" in dimension_info:
                    entities_list = dimension_info["实体"]
                else:
                    # 跳过缺少实体列表的项
                    print(f"警告: 跳过缺少实体列表的维度: {dimension}")
                    entities_list = []
                
                # 添加维度节点并连接到根节点
                # 将描述作为节点属性
                G.add_node(dimension, description=dimension_desc)
                G.add_edge(root, dimension)
                dimension_list.append(dimension)
                
                # 添加规范化的实体节点并连接到维度节点
                for entity in entities_list:
                    # 检查是否已有规范形式
                    if entity in entity_registry.canonical_forms:
                        canonical_entity = entity_registry.canonical_forms[entity]
                    else:
                        # 尝试确定实体类别
                        entity_category = None
                        for category in entity_registry.entities:
                            if entity in entity_registry.entities[category]:
                                entity_category = category
                                break
                        
                        # 添加到实体注册表
                        if entity_category:
                            canonical_entity = entity_registry.add_entity(entity_category, entity)
                        else:
                            # 无法确定类别，使用原始实体
                            canonical_entity = entity
                    
                    G.add_node(canonical_entity)
                    G.add_edge(dimension, canonical_entity)
            
            # 将维度列表作为图的属性保存
            G.graph['dimensions'] = dimension_list
            
            return G
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"尝试解析的JSON: {tree_json[:200]}...")  # 打印部分JSON以便调试
    
    # 创建默认的胜任力树
    return create_default_tree(entities, entity_type)

def create_default_dimensions(entities):
    """创建默认的维度信息"""
    default_dims = []
    
    # 创建技术能力维度
    tech_entities = []
    for category, entity_list in entities.items():
        if category in ["skills", "技术能力", "专业技能"]:
            tech_entities.extend(entity_list)
    if tech_entities:
        default_dims.append({
            "name": "技术能力",
            "description": "技术能力相关的能力和资质",
            "entities": tech_entities
        })
    
    # 创建工作经验维度
    exp_entities = []
    for category, entity_list in entities.items():
        if category in ["experience", "工作经验", "项目经验"]:
            exp_entities.extend(entity_list)
    if exp_entities:
        default_dims.append({
            "name": "工作经验",
            "description": "工作经验相关的能力和资质",
            "entities": exp_entities
        })
    
    # 创建教育背景维度
    edu_entities = []
    for category, entity_list in entities.items():
        if category in ["education", "教育背景", "学历"]:
            edu_entities.extend(entity_list)
    if edu_entities:
        default_dims.append({
            "name": "教育背景",
            "description": "教育背景相关的能力和资质",
            "entities": edu_entities
        })
    
    # 如果没有任何默认维度，创建一个通用维度
    if not default_dims:
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        default_dims.append({
            "name": "通用能力",
            "description": "综合能力和资质",
            "entities": all_entities
        })
    
    return default_dims

def create_default_tree(entities, entity_type):
    """在API调用失败或JSON解析错误的情况下，创建一个基本的树"""
    print("使用默认方式创建胜任力树...")
    
    G = nx.DiGraph()
    root = f"{entity_type}_root"
    G.add_node(root)
    
    # 使用默认维度创建树
    default_dimensions = create_default_dimensions(entities)
    dimension_list = []
    
    for dim_info in default_dimensions:
        dim_name = dim_info["name"]
        dim_desc = dim_info["description"]
        entities_list = dim_info["entities"]
        
        G.add_node(dim_name, description=dim_desc)
        G.add_edge(root, dim_name)
        dimension_list.append(dim_name)
        
        # 添加实体
        for entity in entities_list:
            canonical_entity = entity
            # 尝试找出规范形式
            if entity in entity_registry.canonical_forms:
                canonical_entity = entity_registry.canonical_forms[entity]
                
            G.add_node(canonical_entity)
            G.add_edge(dim_name, canonical_entity)
    
    G.graph['dimensions'] = dimension_list
    return G
# =============================================
# 第六部分：知识图谱构建
# =============================================
def build_knowledge_graph(triples):
    """根据三元组构建知识图谱"""
    G = nx.DiGraph()
    
    # 统计关系类型
    relation_types = set()
    for subj, rel, obj in triples:
        relation_types.add(rel)
    
    # 添加节点和边
    for subj, rel, obj in triples:
        # 如果节点不存在则添加
        if not G.has_node(subj):
            G.add_node(subj, type='subject')
        if not G.has_node(obj):
            G.add_node(obj, type='object')
        
        # 添加边，关系作为边的属性
        G.add_edge(subj, obj, relation=rel)
    
    # 将关系类型列表添加为图的属性
    G.graph['relation_types'] = list(relation_types)
    
    return G

def extract_and_build_semantic_model(text, entity_type):
    """提取实体和关系，并构建语义模型，使用缓存加速"""
    # 首先检查缓存
    cached_model = cache_manager.get_cached_model(text, entity_type)
    if cached_model:
        print(f"使用缓存的{entity_type}语义模型")
        return cached_model
    
    print(f"未找到缓存，开始构建{entity_type}语义模型...")
    
    # 以下是原有的处理逻辑
    # 1. 提取实体
    entities = extract_entities_with_llm(text, entity_type)
    
    # 2. 提取关系三元组
    triples = extract_relations_with_llm(entities, entity_type)
    
    # 3. 构建知识图谱
    knowledge_graph = build_knowledge_graph(triples)
    
    # 4. 构建胜任力模型树
    competency_tree = build_competency_tree_with_llm(entities, entity_type)
    
    # 创建综合模型
    model = {
        'entities': entities,
        'triples': triples,
        'knowledge_graph': knowledge_graph,
        'competency_tree': competency_tree
    }
    
    # 保存到缓存
    cache_manager.save_model_to_cache(text, entity_type, model)
    
    return model

# =============================================
# 第七部分：语义匹配计算（纯Python实现）
# =============================================
def calculate_semantic_matching(resume_model, jd_model):
    """使用Python直接计算简历和职位的语义匹配度"""
    # 1. 提取实体和关系
    resume_entities_dict = resume_model['entities']
    jd_entities_dict = jd_model['entities']
    
    resume_tree = resume_model['competency_tree']
    jd_tree = jd_model['competency_tree']
    
    resume_triples = resume_model['triples']
    jd_triples = jd_model['triples']
    
    # 2. 计算维度覆盖度
    def calculate_dimension_coverage(resume_tree, jd_tree):
        """计算维度覆盖度"""
        resume_dims = set([node for u, node in resume_tree.edges() if u == "resume_root"])
        jd_dims = set([node for u, node in jd_tree.edges() if u == "jd_root"])
        
        # 计算各维度的语义相似度
        dim_matches = 0
        for jd_dim in jd_dims:
            best_match_score = 0
            jd_dim_desc = jd_tree.nodes[jd_dim].get('description', '')
            
            for resume_dim in resume_dims:
                resume_dim_desc = resume_tree.nodes[resume_dim].get('description', '')
                
                # 计算维度名称匹配度
                if jd_dim.lower() == resume_dim.lower():
                    name_similarity = 1.0
                elif jd_dim.lower() in resume_dim.lower() or resume_dim.lower() in jd_dim.lower():
                    name_similarity = 0.7
                else:
                    # 简单词汇重叠计算
                    jd_words = set(jd_dim.lower().split())
                    resume_words = set(resume_dim.lower().split())
                    if jd_words and resume_words:
                        common_words = jd_words.intersection(resume_words)
                        name_similarity = len(common_words) / max(len(jd_words), len(resume_words))
                    else:
                        name_similarity = 0
                
                # 计算描述匹配度
                desc_similarity = 0
                if jd_dim_desc and resume_dim_desc:
                    jd_desc_words = set(jd_dim_desc.lower().split())
                    resume_desc_words = set(resume_dim_desc.lower().split())
                    if jd_desc_words and resume_desc_words:
                        common_words = jd_desc_words.intersection(resume_desc_words)
                        desc_similarity = len(common_words) / max(len(jd_desc_words), len(resume_desc_words))
                
                # 计算综合匹配度
                similarity = 0.7 * name_similarity + 0.3 * desc_similarity
                best_match_score = max(best_match_score, similarity)
            
            # 如果最佳匹配超过阈值，认为该维度被覆盖
            if best_match_score >= 0.3:
                dim_matches += best_match_score
                
        coverage = dim_matches / len(jd_dims) if len(jd_dims) > 0 else 0
        return coverage
    
    # 3. 计算实体匹配度
    def calculate_entity_matching(resume_entities_dict, jd_entities_dict):
        """计算实体匹配度"""
        match_scores = []
        
        # 遍历所有实体类别
        for category in set(resume_entities_dict.keys()) | set(jd_entities_dict.keys()):
            # 跳过空类别
            if (category not in resume_entities_dict or not resume_entities_dict[category]) or \
               (category not in jd_entities_dict or not jd_entities_dict[category]):
                continue
                
            resume_entities = set(resume_entities_dict.get(category, []))
            jd_entities = set(jd_entities_dict.get(category, []))
            
            if not jd_entities:
                continue
            
            entity_matches = 0
            for jd_entity in jd_entities:
                best_match = 0
                for resume_entity in resume_entities:
                    if jd_entity.lower() == resume_entity.lower():
                        best_match = 1.0
                        break
                    elif jd_entity.lower() in resume_entity.lower() or resume_entity.lower() in jd_entity.lower():
                        best_match = max(best_match, 0.7)
                    else:
                        # 计算词汇重叠
                        jd_words = set(jd_entity.lower().split())
                        resume_words = set(resume_entity.lower().split())
                        if jd_words and resume_words:
                            common_words = jd_words.intersection(resume_words)
                            similarity = len(common_words) / max(len(jd_words), len(resume_words))
                            best_match = max(best_match, similarity)
                
                entity_matches += best_match
            
            if jd_entities:
                score = entity_matches / len(jd_entities)
                match_scores.append((category, score))
        
        # 计算总体实体匹配度
        if match_scores:
            # 默认权重
            default_weight = 1.0 / len(match_scores)
            
            # 特定类别权重
            category_weights = {
                "skills": 0.5,
                "experience": 0.3,
                "education": 0.2,
                "certificates": 0.2,
                "languages": 0.2,
                "requirements": 0.4,
                "responsibilities": 0.3
            }
            
            # 加权平均
            total_weight = 0
            weighted_sum = 0
            for category, score in match_scores:
                weight = category_weights.get(category, default_weight)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0
        else:
            return 0
    
    # 4. 计算关系匹配度
    def calculate_relation_matching(resume_triples, jd_triples):
        """计算关系匹配度"""
        if not jd_triples:
            return 0
        
        # 对三元组按关系类型分组
        resume_relations = {}
        for subj, rel, obj in resume_triples:
            if rel not in resume_relations:
                resume_relations[rel] = []
            resume_relations[rel].append((subj, obj))
        
        jd_relations = {}
        for subj, rel, obj in jd_triples:
            if rel not in jd_relations:
                jd_relations[rel] = []
            jd_relations[rel].append((subj, obj))
        
        # 计算关系类型匹配度
        relation_matches = 0
        for jd_rel, jd_tuples in jd_relations.items():
            best_match_score = 0
            
            # 找到最匹配的简历关系类型
            for resume_rel, resume_tuples in resume_relations.items():
                # 关系名称相似度
                if jd_rel.lower() == resume_rel.lower():
                    rel_similarity = 1.0
                elif jd_rel.lower() in resume_rel.lower() or resume_rel.lower() in jd_rel.lower():
                    rel_similarity = 0.7
                else:
                    # 词汇重叠
                    jd_words = set(jd_rel.lower().split())
                    resume_words = set(resume_rel.lower().split())
                    if jd_words and resume_words:
                        common_words = jd_words.intersection(resume_words)
                        rel_similarity = len(common_words) / max(len(jd_words), len(resume_words))
                    else:
                        rel_similarity = 0
                
                # 如果关系相似度达到阈值，计算对象匹配度
                if rel_similarity >= 0.3:
                    # 对象匹配计数
                    obj_matches = 0
                    for _, jd_obj in jd_tuples:
                        has_match = False
                        for _, resume_obj in resume_tuples:
                            if jd_obj.lower() == resume_obj.lower():
                                has_match = True
                                break
                            elif jd_obj.lower() in resume_obj.lower() or resume_obj.lower() in jd_obj.lower():
                                has_match = True
                                break
                        
                        if has_match:
                            obj_matches += 1
                    
                    # 对象匹配率
                    obj_match_rate = obj_matches / len(jd_tuples) if jd_tuples else 0
                    
                    # 综合得分
                    match_score = 0.4 * rel_similarity + 0.6 * obj_match_rate
                    
                    best_match_score = max(best_match_score, match_score)
            
            relation_matches += best_match_score
        
        return relation_matches / len(jd_relations)
    
    # 5. 计算综合匹配度
    dimension_coverage = calculate_dimension_coverage(resume_tree, jd_tree)
    entity_matching = calculate_entity_matching(resume_entities_dict, jd_entities_dict)
    relation_matching = calculate_relation_matching(resume_triples, jd_triples)
    
    # 记录匹配信息
    print(f"维度匹配: {dimension_coverage:.4f}")
    print(f"实体匹配: {entity_matching:.4f}")
    print(f"关系匹配: {relation_matching:.4f}")
    
    # 6. 识别缺失的关键技能和优势
    missing_skills = []
    strengths = []
    
    # 识别缺失技能
    for category in jd_entities_dict:
        if "skill" in category.lower() or category.lower() in ["技能", "技术能力"]:
            jd_skills = set(jd_entities_dict[category])
            resume_skills = set()
            
            # 收集简历中的所有技能
            for resume_cat in resume_entities_dict:
                if "skill" in resume_cat.lower() or resume_cat.lower() in ["技能", "技术能力"]:
                    resume_skills.update(resume_entities_dict[resume_cat])
            
            for skill in jd_skills:
                has_similar = False
                for resume_skill in resume_skills:
                    if skill.lower() == resume_skill.lower() or skill.lower() in resume_skill.lower() or resume_skill.lower() in skill.lower():
                        has_similar = True
                        break
                if not has_similar:
                    missing_skills.append(skill)
    
    # 识别优势
    for category in resume_entities_dict:
        resume_entities = set(resume_entities_dict[category])
        
        # 检查每个实体在职位要求中是否出现
        for entity in resume_entities:
            is_strength = False
            for jd_category in jd_entities_dict:
                jd_entities = set(jd_entities_dict[jd_category])
                for jd_entity in jd_entities:
                    if entity.lower() == jd_entity.lower() or entity.lower() in jd_entity.lower() or jd_entity.lower() in entity.lower():
                        is_strength = True
                        break
                if is_strength:
                    break
            
            if is_strength and entity not in strengths:
                strengths.append(entity)
    
    if missing_skills:
        print(f"缺少技能: {', '.join(missing_skills[:5])}" + (f"等{len(missing_skills)}项" if len(missing_skills) > 5 else ""))
    
    if strengths:
        print(f"简历优势: {', '.join(strengths[:5])}" + (f"等{len(strengths)}项" if len(strengths) > 5 else ""))
    
    # 7. 加权计算综合分数
    weights = [0.3, 0.5, 0.2]  # 维度覆盖度、实体匹配度、关系匹配度的权重
    overall_matching = weights[0] * dimension_coverage + weights[1] * entity_matching + weights[2] * relation_matching
    
    print(f"综合匹配: {overall_matching:.4f}")
    
    # 8. 组织返回结果
    matching_result = {
        "dimension_coverage": dimension_coverage,
        "entity_matching": entity_matching, 
        "relation_matching": relation_matching,
        "overall_matching": overall_matching,
        "missing_skills": missing_skills,
        "strengths": strengths
    }
    
    return matching_result

# =============================================
# 第八部分：排序与评估
# =============================================
def rank_jobs_for_user(user_id, user_df, jd_df, action_df):
    """为用户排序职位，并显示详细的进度提示，包含列名检查和错误处理"""
    print(f"\n开始为用户 {user_id} 分析匹配职位...")
    
    # 检查数据框的列名
    print("检查数据框列名...")
    
    # 打印各数据框的列名，便于调试
    print(f"用户数据框列名: {list(user_df.columns)}")
    print(f"职位数据框列名: {list(jd_df.columns)}")
    print(f"行为数据框列名: {list(action_df.columns)}")
    
    # 定义列名映射，处理可能的名称不一致问题
    # 用户表列名映射
    user_columns_map = {
        'user_id': ['user_id', 'userid', 'id', 'user'],
        'experience': ['experience', 'work_experience', 'exp'],
        'desire_jd_industry_id': ['desire_jd_industry_id', 'desire_industry', 'industry_desired'],
        'desire_jd_type_id': ['desire_jd_type_id', 'desire_job_type', 'job_type_desired'],
        'cur_industry_id': ['cur_industry_id', 'current_industry', 'industry'],
        'cur_jd_type': ['cur_jd_type', 'current_job_type', 'job_type']
    }
    
    # 职位表列名映射
    jd_columns_map = {
        'jd_no': ['jd_no', 'jdno', 'job_id', 'jobid', 'id'],
        'jd_title': ['jd_title', 'job_title', 'title', 'position_name'],
        'job_description': ['job_description', 'description', 'desc', 'job_desc', 'requirements']
    }
    
    # 行为表列名映射
    action_columns_map = {
        'user_id': ['user_id', 'userid', 'id', 'user'],
        'jd_no': ['jd_no', 'jdno', 'job_id', 'jobid', 'id']
    }
    
    # 创建实际映射字典，用于替换列名
    actual_user_columns = {}
    actual_jd_columns = {}
    actual_action_columns = {}
    
    # 检查用户表
    for std_col, possible_cols in user_columns_map.items():
        for col in possible_cols:
            if col in user_df.columns:
                actual_user_columns[std_col] = col
                break
    
    # 检查职位表
    for std_col, possible_cols in jd_columns_map.items():
        for col in possible_cols:
            if col in jd_df.columns:
                actual_jd_columns[std_col] = col
                break
    
    # 检查行为表
    for std_col, possible_cols in action_columns_map.items():
        for col in possible_cols:
            if col in action_df.columns:
                actual_action_columns[std_col] = col
                break
    
    # 检查是否找到了必要的列
    required_user_cols = ['user_id']
    required_jd_cols = ['jd_no', 'job_description']
    required_action_cols = ['user_id', 'jd_no']
    
    missing_user_cols = [col for col in required_user_cols if col not in actual_user_columns]
    missing_jd_cols = [col for col in required_jd_cols if col not in actual_jd_columns]
    missing_action_cols = [col for col in required_action_cols if col not in actual_action_columns]
    
    if missing_user_cols:
        print(f"错误：用户数据框缺少必要列: {missing_user_cols}")
        print(f"可用列: {list(user_df.columns)}")
        return [], []
    
    if missing_jd_cols:
        print(f"错误：职位数据框缺少必要列: {missing_jd_cols}")
        print(f"可用列: {list(jd_df.columns)}")
        return [], []
    
    if missing_action_cols:
        print(f"错误：行为数据框缺少必要列: {missing_action_cols}")
        print(f"可用列: {list(action_df.columns)}")
        return [], []
    
    print("列名检查完成，开始提取用户信息...")
    
    # 使用映射后的列名获取用户信息
    user_id_col = actual_user_columns['user_id']
    user_info = user_df[user_df[user_id_col] == user_id]
    
    if len(user_info) == 0:
        print(f"错误：找不到用户 {user_id} 的信息")
        return [], []
        
    user_info = user_info.iloc[0]
    
    # 提取用户简历信息
    resume_text = ""
    fields_extracted = []
    
    # 使用映射后的列名提取信息
    for std_col, possible_cols in user_columns_map.items():
        if std_col == 'user_id':
            continue  # 跳过用户ID
            
        actual_col = actual_user_columns.get(std_col)
        if actual_col and actual_col in user_info:
            resume_text += str(user_info[actual_col]) + " "
            fields_extracted.append(std_col)
    
    if fields_extracted:
        print(f"已提取用户信息字段: {', '.join(fields_extracted)}")
    else:
        print(f"警告：未能从用户 {user_id} 提取到有效信息字段")
    
    # 预处理简历文本
    print("正在预处理简历文本...")
    resume_text = preprocess_text(resume_text)
    print(f"简历文本长度: {len(resume_text)} 字符")
    
    # 构建简历的语义模型
    print("正在构建用户简历的语义模型...")
    resume_model = extract_and_build_semantic_model(resume_text, "resume")
    print("简历语义模型构建完成！")
    
    # 获取该用户的候选职位
    print("正在获取候选职位列表...")
    user_id_action_col = actual_action_columns['user_id']
    jd_no_action_col = actual_action_columns['jd_no']
    
    candidate_jobs = action_df[action_df[user_id_action_col] == user_id][jd_no_action_col].unique()
    job_count = len(candidate_jobs)
    print(f"找到 {job_count} 个候选职位需要计算匹配度")
    
    if job_count == 0:
        print(f"警告：用户 {user_id} 没有候选职位，无法进行匹配计算")
        return [], []
    
    # 计算每个职位的匹配分数
    job_scores = []
    job_details = []
    
    # 获取职位表的必要列名
    jd_no_col = actual_jd_columns['jd_no']
    jd_desc_col = actual_jd_columns['job_description']
    jd_title_col = actual_jd_columns.get('jd_title', '')  # 标题是可选的
    
    print("\n开始计算职位匹配度...")
    for i, jd_no in enumerate(candidate_jobs, 1):
        progress_percent = (i / job_count) * 100
        print(f"正在处理第 {i}/{job_count} 个职位 (ID: {jd_no})... {progress_percent:.1f}%")
        
        # 获取职位信息
        job_info = jd_df[jd_df[jd_no_col] == jd_no]
        if len(job_info) == 0:
            print(f"  警告：找不到职位 {jd_no} 的详细信息，已跳过")
            continue
            
        # 确保职位描述列存在
        if jd_desc_col not in job_info.columns:
            print(f"  错误：职位 {jd_no} 缺少描述列 '{jd_desc_col}'")
            continue
            
        job_desc = str(job_info.iloc[0][jd_desc_col]) if not pd.isna(job_info.iloc[0][jd_desc_col]) else ""
        
        # 获取职位标题，如果标题列存在
        job_title = "未知职位"
        if jd_title_col and jd_title_col in job_info.columns:
            job_title = str(job_info.iloc[0][jd_title_col]) if not pd.isna(job_info.iloc[0][jd_title_col]) else "未知职位"
        
        print(f"  开始分析职位: {job_title}")
        
        # 预处理职位描述
        job_desc = preprocess_text(job_desc)
        desc_len = len(job_desc)
        print(f"  职位描述长度: {desc_len} 字符")
        
        if desc_len < 10:
            print(f"  警告：职位 {jd_no} 描述过短，可能影响匹配质量")
            continue  # 跳过描述过短的职位
        
        # 构建职位的语义模型
        print(f"  正在构建职位语义模型...")
        jd_model = extract_and_build_semantic_model(job_desc, "jd")
        print(f"  职位语义模型构建完成！")
        
        # 计算语义匹配分数和详情
        print(f"  计算语义匹配度...")
        match_result = calculate_semantic_matching(resume_model, jd_model)
        score = match_result["overall_matching"]
        print(f"  匹配计算完成！匹配度: {score:.4f}")
        
        # 记录分数
        job_scores.append((jd_no, score))
        
        # 记录详情
        missing_skills = match_result["missing_skills"][:5] if match_result["missing_skills"] else []
        strengths = match_result["strengths"][:5] if match_result["strengths"] else []
        
        job_details.append({
            "jd_no": jd_no,
            "title": job_title,
            "score": score,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "dimension_score": match_result["dimension_coverage"],
            "entity_score": match_result["entity_matching"],
            "relation_score": match_result["relation_matching"]
        })
        
        if missing_skills:
            print(f"  发现缺少技能: {', '.join(missing_skills)}")
        if strengths:
            print(f"  发现匹配优势: {', '.join(strengths)}")
    
    print(f"\n所有 {job_count} 个职位分析完成！正在进行排序...")
    
    # 按分数降序排序
    ranked_jobs = sorted(job_scores, key=lambda x: x[1], reverse=True)
    ranked_details = sorted(job_details, key=lambda x: x["score"], reverse=True)
    
    # 输出排序结果的详细信息
    print(f"\n用户 {user_id} 的职位排序结果:")
    print(f"=" * 50)
    
    if not ranked_details:
        print("没有找到匹配的职位")
        return [], []
        
    top_score = ranked_details[0]["score"] if ranked_details else 0
    
    for i, job in enumerate(ranked_details[:5], 1):
        relative_score = (job['score'] / top_score) * 100 if top_score > 0 else 0
        
        print(f"{i}. 职位: {job['title']} (ID: {job['jd_no']})")
        print(f"   匹配分数: {job['score']:.4f} ({relative_score:.1f}% 相对于最高分)")
        print(f"   维度匹配: {job['dimension_score']:.4f} | 实体匹配: {job['entity_score']:.4f} | 关系匹配: {job['relation_score']:.4f}")
        
        if job['missing_skills']:
            print(f"   缺少技能: {', '.join(job['missing_skills'])}")
        if job['strengths']:
            print(f"   匹配优势: {', '.join(job['strengths'])}")
        print(f"   {'-' * 40}")
    
    # 显示汇总信息
    if len(ranked_details) > 5:
        print(f"还有 {len(ranked_details)-5} 个匹配职位未显示")
    
    print(f"=" * 50)
    
    return [job[0] for job in ranked_jobs], ranked_details  # 返回排序后的职位ID列表和详情

def calculate_MAP(ranked_lists, action_df, target_column):
    """计算MAP指标"""
    ap_values = []
    
    for user_id, ranked_jobs in ranked_lists.items():
        # 获取该用户的真实职位交互记录
        user_actions = action_df[action_df['user_id'] == user_id]
        
        # 获取正例（投递或中意的职位）
        positive_jobs = user_actions[user_actions[target_column] == 1]['jd_no'].tolist()
        
        if len(positive_jobs) == 0:
            continue  # 跳过没有正例的用户
        
        # 计算AP
        precision_sum = 0
        pos_count = 0
        
        for k, job_id in enumerate(ranked_jobs, 1):
            if job_id in positive_jobs:
                pos_count += 1
                precision_at_k = pos_count / k
                precision_sum += precision_at_k
        
        ap = precision_sum / len(positive_jobs) if len(positive_jobs) > 0 else 0
        ap_values.append(ap)
    
    # 计算MAP
    map_value = sum(ap_values) / len(ap_values) if len(ap_values) > 0 else 0
    return map_value

def evaluate_model(user_df, jd_df, action_df, test_user_ids):
    """评估模型效果"""
    ranked_lists = {}
    all_job_details = {}
    
    for user_id in test_user_ids:
        print(f"正在为用户 {user_id} 排序职位...")
        ranked_jobs, job_details = rank_jobs_for_user(user_id, user_df, jd_df, action_df)
        ranked_lists[user_id] = ranked_jobs
        all_job_details[user_id] = job_details
        print(f"用户 {user_id} 的职位排序完成，共 {len(ranked_jobs)} 个职位\n")
    
    # 计算投递的MAP
    map_delivered = calculate_MAP(ranked_lists, action_df, 'delivered')
    
    # 计算中意的MAP
    map_satisfied = calculate_MAP(ranked_lists, action_df, 'satisfied')
    
    # 计算最终评价值
    map_final = map_satisfied * 0.7 + map_delivered * 0.3
    
    return map_delivered, map_satisfied, map_final, all_job_details

# 在程序退出时保存实体注册表
def save_entity_registry_on_exit():
    if 'entity_registry' in globals() and 'cache_manager' in globals():
        print("程序退出，正在保存实体注册表...")
        cache_manager.save_entity_registry_to_cache(entity_registry)

# 注册退出时执行的函数
atexit.register(save_entity_registry_on_exit)

# =============================================
# 第九部分：主函数
# =============================================
def main():
    # 加载数据
    print("正在加载数据...")
    user_df, jd_df, action_df = load_data()
    
    # 显示缓存状态
    resume_cache_files = len(os.listdir(cache_manager.resume_cache_dir))
    jd_cache_files = len(os.listdir(cache_manager.jd_cache_dir))
    print(f"当前缓存状态: {resume_cache_files}个简历模型, {jd_cache_files}个职位模型")
    
    # 尝试从缓存加载实体注册表
    global entity_registry
    cached_registry = cache_manager.get_cached_entity_registry()
    if cached_registry:
        entity_registry = cached_registry
        print("已从缓存加载实体注册表")
        # 显示已加载的实体类别
        for category, entities in entity_registry.entities.items():
            print(f"- {category}: {len(entities)}个实体")
    else:
        # 初始化实体清单
        print("正在初始化实体清单...")
        initialize_entity_registry(user_df, jd_df)
        print(f"实体清单初始化完成，当前实体类别:")
        for category, entities in entity_registry.entities.items():
            print(f"- {category}: {len(entities)}个实体")
    
    # 检查API调用是否可用
    try:
        # 简单测试API可用性
        test_response = call_gpt_api("测试API连接。请回复'连接成功'。")
        if test_response and "连接成功" in test_response:
            print("API连接测试成功，gpt-4o-mini模型可用")
        else:
            print("警告: API连接测试未返回预期结果，请检查API配置")
            if test_response:
                print(f"API返回: {test_response[:100]}...")
            return
    except Exception as e:
        print(f"无法连接到API: {e}")
        print("请检查API地址和令牌是否正确")
        return
    
    print("API连接和gpt-4o-mini模型已就绪，开始执行招聘推荐系统...")
    
    # 划分训练集和测试集
    all_user_ids = user_df['user_id'].unique()
    np.random.seed(42)  # 设置随机种子，确保结果可复现
    test_ratio = 0.3
    test_size = int(len(all_user_ids) * test_ratio)
    
    # 为了测试效率，这里只使用少量数据
    test_user_ids = np.random.choice(all_user_ids, min(test_size, 3), replace=False)
    
    print(f"使用{len(test_user_ids)}个用户进行测试...")
    
    # 评估模型
    print("开始评估模型...")
    map_delivered, map_satisfied, map_final, job_details = evaluate_model(user_df, jd_df, action_df, test_user_ids)
    
    print(f"\n评估结果:")
    print(f"投递职位MAP: {map_delivered:.4f}")
    print(f"中意职位MAP: {map_satisfied:.4f}")
    print(f"最终评价值MAP: {map_final:.4f}")
    
    # 在评估模型后，清理没有实体的类别
    print("\n开始清理空的实体类别...")
    removed_count = entity_registry.clean_empty_categories(min_entities=1)
    print(f"共清理了{removed_count}个空类别")
    
    # 输出实体清单统计
    print("\n实体清单最终状态:")
    for category, entities in entity_registry.entities.items():
        print(f"- {category}: {len(entities)}个实体")
    
    # 输出新发现的类别
    initial_categories = ["技能", "工作经验", "教育背景", "行业领域", "职位要求", "工作职责"]
    new_categories = [cat for cat in entity_registry.entities.keys() if cat not in initial_categories]
    
    if new_categories:
        print("\n系统自动发现的新实体类别:")
        for category in new_categories:
            print(f"- {category}: {entity_registry.category_descriptions.get(category, '无描述')} ({len(entity_registry.entities[category])}个实体)")
            # 打印该类别中的前20个实体（或全部，如果少于20个）
            entity_list = list(entity_registry.entities[category].keys())
            display_entities = entity_list[:20]
            if entity_list:
                print(f"  实体列表: {', '.join(display_entities)}" + 
                    (f" 等共{len(entity_list)}个" if len(entity_list) > 20 else ""))
    
    # 在程序结束前保存实体注册表到缓存
    cache_manager.save_entity_registry_to_cache(entity_registry)
    print("已保存实体注册表到缓存")
    
    return map_final, job_details

if __name__ == "__main__":
    main()