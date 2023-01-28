from setuptools import setup, find_packages


setup(
	name="Fau_tools",
	version="1.3.5",
	author="utility",
	author_email="Fau818@qq.com",
	url="https://github.com/Fau818/Fau_tools",
	license="MIT",
	description="A python module. The main function is for pytorch training.",
	python_requires=">=3.6",
	packages=find_packages(),
	package_data={"": ["*"]},  # 数据文件全部打包
)
