from setuptools import setup, find_packages

setup(
    # 1. 定义您的项目名称。请使用一个与标准 'transformers' 不同的名称，
    #    以避免将来与 pip 安装的标准版本冲突。
    #    这里我们使用 'custom-transformers' 作为示例。
    name='custom-transformers',
    
    # 2. 定义项目的版本号
    version='4.99.0.dev0', # 可以使用一个较高的版本号
    
    # 3. 描述
    description='A local, customized version of the Hugging Face Transformers library.',
    
    # 4. 作者信息
    author='Your Name',
    
    # 5. 许可证
    license='Apache License 2.0', # 保持与原库一致的许可证
    
    # 6. 指定要安装的包
    # 
    # **关键点：** 'packages' 字段定义了哪些文件夹应该被视为可导入的包。
    # 因为您本地的项目中只有 'transformers' 文件夹，我们手动指定它。
    # 注意：这里我们告诉 setuptools 我们的包是 'transformers'。
    packages=['transformers'],
    
    # 7. 自动查找子包（可选，但推荐）
    # find_packages() 会自动在当前目录下查找所有带有 __init__.py 的子目录作为包。
    # 如果您只有 transformers 文件夹，使用 ['transformers'] 更明确。
    
    # 8. 依赖关系（非常重要！）
    # 
    # 您需要列出 transformers 库本身依赖的所有外部库。
    # 这是一个简化列表，您应该根据您的代码修改版本所需的依赖进行调整。
    # 如果您不确定，可以查看标准 transformers 仓库的 setup.py。
    install_requires=[
        "numpy",
        "requests",
        "filelock",
        "tqdm",
        "safetensors",
        "pyyaml",
        "huggingface-hub>=0.19.0", 
        # 其他依赖项，例如：
        # "torch>=1.13.0",
        # "sentencepiece>=0.1.99"
    ],
    
    # 9. Python 版本要求
    python_requires='>=3.8',
)