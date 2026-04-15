import os, re
path = 'eval/generate_testset.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

c = re.sub(
    r'    test_size = 50 .*?    output_path = \"eval/datasets/', 
    r'''    test_size = 50 # Baseline size
    print(f"🚀 generating {test_size} questions with gemini-3/3.1")
    
    os.environ["RAGAS_MAX_RETRIES"] = "3"

    # Ragas 0.3.x 接口
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        transforms_llm=critic_llm,
        with_debugging_logs=True
    )

    df = testset.to_pandas()
    output_path = \"eval/datasets/''', 
    c, flags=re.DOTALL
)

c = re.sub(
    r'    distributions = \{.*?\}\n', '', c, flags=re.DOTALL
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

