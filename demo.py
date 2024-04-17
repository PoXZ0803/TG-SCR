#Haikou & 2010~2020 | Shanghai | Qingdao  | Hanzghou.
from search_insql import search_time,age_to_year,geo_to_year,main_search
from tree_search_in_sql import tree_search_in_sql
from bouding import bounding_box
from extract import extract
from keywordextract import split_str
import pandas as pd
import pymysql
import datetime
import pandas as pd
import time as tm
import argparse
conn = pymysql.connect(
    host='server.acemap.cn',
    user='readonly',
    password='readonly',
    database='md_paper',
    port=13306
)
cursor = conn.cursor()
# def parse_args():
#     parse = argparse.ArgumentParser()
#     parse.add_argument('--flag',default=0)
#     parse.add_argument('--fuzzymonth',default=0)
#     parse.add_argument('--latitude_range',default=(20, 25))
#     parse.add_argument('--longitude_range',default=(32, 35))
#     parse.add_argument('--km',default=10)
#     args = parse.parse_args()
#     return args
# args = parse_args()
def search_in_sentence():
    sentence = input("请输入问句:")
    start,end,city = extract(sentence)
    if len(city):
        city = city[0]
    # fuzzy_search_flag = args.flag
    # fuzzy_search_time = args.fuzzymonth
    # latitude_range = args.latitude_range
    # longitude_range = args.longitude_range
    # km = args.km
    starts = start.split('.')
    ends = end.split('.')
    start = starts[0]+'.'+starts[1]
    end = ends[0]+'.'+ends[1]
    startime =tm.time() 
    paper_index_1 = main_search(start,end)
    if len(city):
        paper_index_2 = bounding_box(name=city)
        paper_index_3 = tree_search_in_sql(city)
    else:
        paper_index_2=[]
        paper_index_3=[]
    need_index = list(set(paper_index_3).union(set(paper_index_2)))
    if len(need_index) != 0 and len(city):
        need_index = list(set(need_index)&set(paper_index_1))
    else:
        need_index = paper_index_1
    endtime = tm.time()
    # print('wewe',paper_index_1)
    # print('wewe',paper_index_2)
    print("knowledge paper_id:",need_index)
    usedtime = endtime - startime
    usedtime = int(usedtime*1000)/1000
    print("time used:" ,usedtime,"s") 
    results = []
    tm.sleep(2)
    for i in range(len(need_index)):
        query = f"SELECT * FROM {'search_tree_db'} WHERE paper_id = '{need_index[i]}'"
        cursor.execute(query)
        results.append(cursor.fetchone())
    for result in results:
        print(results)
def key_word_search():
    sentence = input("请输入关键词:")
    method = input("请输入关键词搜索方式(支持location，time，both)：")
    time_result, space_result,_ = split_str(input = sentence)
    startime =tm.time()
    if method == "time":
        final_list = []
        for time in time_result.keys():
            timed = time.split('~')
            if len(timed) == 1:
                times = time.split('.')
                year = times[0]
                if len(times) > 1:
                    month = times[1]
                    if month.startswith('0') and len(month) > 1:
                        month = month[1:]
                else:
                    month = '0'
                paper_index_1 = search_time(year,month)
            else:
                start = timed[0]
                end = timed[1]
                start =start.split('.')[0]
                end = end.split('.')[0]
                paper_index_1 = main_search(start,end)
            if len(paper_index_1) == 0:
                print('未找到准确时间数据，输出：',year,'年数据')
                paper_index_1 = search_time(year)
            if time_result[time] == '|':
                final_list = list(set(paper_list).union(set(paper_index_1)))
            elif time_result[time] =='&':
                final_list = list(set(paper_list).intersection(set(paper_index_1)))
            else:
                final_list = paper_index_1
    elif method == 'location':
        final_list = []
        for space in space_result.keys():
            city = space
            paper_index_2 = bounding_box(name=city)
            paper_index_3 = tree_search_in_sql(city)
            paper_index_1 = list(set(paper_index_3).union(set(paper_index_2)))
            if space_result[space] == '|':
                final_list = list(set(final_list).union(set(paper_index_1)))
            elif space_result[space] =='&':
                final_list = list(set(final_list).intersection(set(paper_index_1)))
            else:
                final_list = paper_index_1
    elif method == 'both':
        space_paper_list =[]
        time_paper_list = []
        i = 0
        j = 0
        flag = ''
        for time in time_result.keys():
            timed = time.split('~')
            if len(timed) == 1:
                times = time.split('.')
                year = times[0]
                if len(times) > 1:
                    month = times[1]
                    if month.startswith('0') and len(month) > 1:
                        month = month[1:]
                    paper_index_1 = search_time(year,month)
                else:
                    paper_index_1 = search_time(year)
                if len(paper_index_1) == 0:
                    print('未找到准确时间数据，输出：',year,'年数据')
                    paper_index_1 = search_time(year)
            else:
                start = timed[0]
                end = timed[1]
                start =start.split('.')[0]
                end = end.split('.')[0]
                paper_index_1 = main_search(start,end)
            if i == 0 :
                time_paper_list = paper_index_1
                if time_result[time] != '':
                    flag = time_result[time]
                    i = i + 1
                continue
            if time_result[time] == '|':
                time_paper_list = list(set(time_paper_list).union(set(paper_index_1)))
            elif time_result[time] =='&':
                time_paper_list = list(set(time_paper_list).intersection(set(paper_index_1)))
            i = i+1
        for space in space_result.keys():
            city = space
            paper_index_2 = bounding_box(name=city)
            paper_index_3 = tree_search_in_sql(city)
            paper_index_1 = list(set(paper_index_3).union(set(paper_index_2)))
            if j == 0 :
                space_paper_list = paper_index_1
                if space_result[space] != '':
                    flag = space_result[space]
                j = j + 1
                continue
            if space_result[space] == '|':
                space_paper_list = list(set(space_paper_list).union(set(paper_index_1)))
            elif space_result[space] =='&':
                space_paper_list = list(set(space_paper_list).intersection(set(paper_index_1)))          
            j = j+1
        final_list = []
        if flag == '&':
            final_list = list(set(space_paper_list).intersection(set(time_paper_list)))
        elif flag == '|':
            final_list = list(set(space_paper_list).union(set(time_paper_list)))
        elif flag == '':
            final_list = list(set(space_paper_list).union(set(time_paper_list)))
    endtime = tm.time()
    print("knowledge paper_id:",final_list)
    usedtime = endtime - startime
    usedtime = int(usedtime*1000)/1000
    print("time used:" ,usedtime,"s") 
    results = []
    tm.sleep(2)
    for i in range(len(final_list)):
        query = f"SELECT * FROM {'search_tree_db'} WHERE paper_id = '{final_list[i]}'"
        cursor.execute(query)
        results.append(cursor.fetchone())
    for result in results:
        print(results)
if __name__ == "__main__" :
    method = input("请输入搜索方式(支持sentence,key_words)：")
    if method == "sentence":
        search_in_sentence()
    elif method == 'key_words':
        key_word_search()
cursor.close()
conn.close()
