import doc2vec


# traing doc2vec model

get_d2v_model = doc2vec.get_d2v_model

'''
model = get_d2v_model(
    txt_dir_name=图片文件夹路径，
    vec_dimension=产生文档向量的维度
    iter_=算法迭代次数
    min_count=词的最小出现次数
    model_name=希望模型保存输出到当前目录下的文件名
)
'''


query = doc2vec.query

'''
docVec = query(
    d2v_model_name=模型的名称
    txt_file_name=文档的文件路劲

)
'''


if __name__ == "__main__":
    txt_dir_name= "/media/rw/DATA/sogou/formalCompetition4/News_info_train"
    # get_d2v_model(txt_dir_name)
    print(query("d2vModel", "/media/rw/DATA/sogou/formalCompetition4/News_info_train/2016999966.txt"))
    print(query("d2vModel", "/media/rw/DATA/sogou/formalCompetition4/News_info_train/2016999966.txt"))
    print(query("d2vModel", "/media/rw/DATA/sogou/formalCompetition4/News_info_train/2016999966.txt"))
