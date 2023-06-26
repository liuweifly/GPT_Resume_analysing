import streamlit as st
import openai
from PyPDF2 import PdfReader
import io
import csv
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

openai_api_key = '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'
openai.api_key = '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'

# 设置页面宽度为较宽的布局
st.set_page_config(layout="wide")

def analyze_resume(jd, resume, options):
    df = analyze_str(resume, options)
    #print(df)
    df_string = df.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    #print(df_string)
    st.write("OpenAI综合分析..")
    summary_question = f"职位要求是：{{{jd}}}" + f"简历概要是：{{{df_string}}}" + "，请直接返回该应聘岗位候选人匹配度概要（控制在200字以内）;'"
    summary = ask_openAI(summary_question)
    print(summary)

    df.loc[len(df)] = ['综合概要', summary]
    extra_info = "打分要求：国内top10大学+3分，985大学+2分，211大学+1分，头部企业经历+2分，知名企业+1分，海外背景+3分，外企背景+1分。 "
    score_question = f"职位要求是：{{{jd}}}" + f"简历概要是：{{{df.to_string(index=False)}}}" + "，请直接返回该应聘岗位候选人的匹配分数（0-100），请精确打分以方便其他候选人对比排序，'" + extra_info

    score = ask_openAI(score_question)
    df.loc[len(df)] = ['匹配得分', score]

    return df

def ask_openAI(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=800,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text.strip()

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    # Open (or create) a csv file in write mode
    with open('chunks.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["Index", "Chunk"])

        # Write the chunks with their respective indices
        for i, chunk in enumerate(chunks):
            writer.writerow([i, chunk])

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    df_data = [{'option': option, 'value': []} for option in options]
    st.write("信息抓取")

    # 创建进度条和空元素
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="信息抓取中", unit="选项", ncols=100):
        question = f"这个应聘者的{option}是什么，请精简返回答案，最多不超过300字，如果查找不到，则返回'未提供'"
        docs = knowledge_base.similarity_search(question)
        print(len(docs))
        llm = OpenAI(openai_api_key=openai_api_key,temperature=0.1, verbose=False)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        df_data[i]['value'] = response
        option_status.text(f"正在查找信息：{option}")

        # 更新进度条
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(df_data)
    st.success("简历要素已获取")
    return df



def main():

    # 设置页面标题
    st.title("🚀 GPT招聘分析机器人")
    st.subheader("🪢 Langchain + 🎁 OpenAI")

    # 设置默认的JD和简历信息
    default_jd = """高级策略产品经理 3.5万-4.5万  15薪 
        职位描述:
        1、负责C端策略相关产品的设计工作，设计针对不同用户层次的产品承接策略，包括但不限于搜索、推荐、即时消息、推送等场景，并负责用户留存和投递量。
        2、深入洞察招聘行业用户的核心需求场景，与用户研究团队合作，探索并理解用户的行为、偏好和痛点，将这些洞察转化为前端产品形态的设计方向。
        3、与算法策略团队紧密协作，理解推荐系统的基本框架和机器学习的基本原理，提供产品需求和指导，确保产品的推荐策略与用户体验和数据指标的正向发展一致。
        4、与其他业务团队紧密合作，包括但不限于运营、技术和设计团队，共同提升用户体验和流量效率，推动产品的持续优化和创新。
        5、通过数据定量分析、定性观察和用户反馈，深入了解用户行为和产品效果，不断改进产品和策略方案，提高用户满意度和业务成果。
        职位要求：\
        1、本科及以上学历，具备5-10年推荐策略产品经验，有招聘行业经验者优先考虑。
        2、对推荐系统的基本框架和机器学习的基本原理有深入的理解，并能将其应用于产品设计和策略制定中。
        3、具备良好的用户体验感知，了解招聘场景下用户的需求和体验认知，能够将用户洞察转化为实际的产品改进和创新。
        4、数据驱动思维，能够通过数据分析和用户研究来支持产品决策，具备定量和定性分析的能力。
        5、具备出色的项目协作能力和沟通能力，能够高效推进项目，跨团队合作并与各方利益相关者保持良好的沟通。
        6、具备强大的抗压能力和问题解决能力，能够在快节奏的工作环境中处理多任务并保持高质量的工作成果。
        我们希望你能为我们的团队带来创新思维和积极进取的态度，并与我们共同努力提供卓越的用户体验和业务成果。
        如果您对该职位感兴趣，请提交您的简历及相关作品，我们期待与您进一步交流。"""

    # 输入JD信息
    jd_text = st.text_area("【岗位信息】", height=300, value=default_jd)

    # 输入简历信息
    #resume_text = st.text_area("【应聘简历】", height=100, value=default_resume)
    uploaded_file = st.file_uploader("【应聘简历】", type="pdf")
    if uploaded_file is not None:
        pdf = PdfReader(io.BytesIO(uploaded_file.getvalue()))  
        resume_text = ""
        for page in range(len(pdf.pages)):  
            resume_text += pdf.pages[page].extract_text()  
        #st.text(resume_text)
        st.markdown(
            f'<div style="height:300px;overflow:auto;">{resume_text}</div>', 
            unsafe_allow_html=True
        )


    # 参数输入
    options = ["姓名", "联系号码", "性别", "年龄", "工作年数（数字）", "最高学历", "本科学校名称", "硕士学校名称", "是否在职", "当前职务", "历史任职公司列表", "技术能力", "经验程度", "管理能力"]
    selected_options = st.multiselect("请选择选项", options, default=options)

    # 分析按钮
    if st.button("开始分析"):
        df = analyze_resume(jd_text, resume_text, selected_options)
        st.subheader("综合匹配得分："+ df.loc[df['option'] == '匹配得分', 'value'].values[0])
        st.subheader("细项展示：")
        st.table(df)

if __name__ == "__main__":
    main()
