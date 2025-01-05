import csv
import math,random

def read_csv(file_path, user_num):
    """
    读取CSV文件，筛选userId在指定范围内的记录，并处理movieId。

    参数：
    file_path: str, CSV文件路径
    user_num: int, 保留userId范围(从1到user_num)内的记录

    返回：
    list[tuple]: 符合条件的(userId, movieId)元组列表，movieId处理为除以200后向上取整。
    """
    result = []

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            user_id = int(row['userId'])
            if user_id > user_num:
                continue
            movie_id = int(row['movieId'])

            if 1 <= user_id <= user_num:
                processed_movie_id = math.ceil(movie_id / 200)
                result.append((user_id, processed_movie_id))
    random.shuffle(result)
    return result
def read_content_attr():
    """
    读取CSV文件，筛选userId在指定范围内的记录，并处理movieId。

    参数：
    file_path: str, CSV文件路径
    user_num: int, 保留userId范围(从1到user_num)内的记录

    返回：
    list[tuple]: 符合条件的(userId, movieId)元组列表，movieId处理为除以200后向上取整。
    """
    file_path = r'C:\Users\14459\PycharmProjects\dynamics_sim\content_data.csv'
    result = []
    size_dict = {}
    importance_dict = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            c_id = int(row['contentId'])
            c_size = int(row['contentSize'])
            c_importance = int(row['importance'])
            size_dict[c_id] = c_size
            importance_dict[c_id] = c_importance
            
    return size_dict, importance_dict
# file_path = 'movies.csv'
# user_num = 10
# print(process_csv(file_path, user_num))
