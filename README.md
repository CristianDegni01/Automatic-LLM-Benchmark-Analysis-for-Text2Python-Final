# [EURECOM] Semester Project Text2Code: Can LLMs query a database without SQL?
by Cristian DEGNI, Francesco GIANNUZZO
### Abstract: 
In this study, we investigate the ability of Large Language Models to autonomously generate Python code to query databases, a task we refer to as Text-to-Code. We systematically compare this approach with two established methods: Text-to-SQL} and Table Question Answering, evaluating them across multiple domains, query complexities, and model architectures. To support our analysis, we introduce a dedicated framework for generation, execution and evaluation by ad hoc metrics. While Text-to-Code shows advantages in problems requiring complex reasoning or looser adherence to database structure, Text-to-SQL remains superior for strictly SQL-centric queries.

## 📚 References

1. Gilbert Badaro, Mohammed Saeed, and Paolo Papotti. *Transformers for tabular data representation: A survey of models and applications*. Transactions of the Association for Computational Linguistics 11 (2023), 227–249.

2. Peter Baile Chen, Fabian Wenz, Yi Zhang, Devin Yang, Justin Choi, Nesime Tatbul, Michael Cafarella, Çağatay Demiralp, and Michael Stonebraker. *BEAVER: an enterprise benchmark for text-to-sql*. arXiv preprint arXiv:2409.02038 (2024).

3. Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. *Binding Language Models in Symbolic Languages*. arXiv preprint arXiv:2210.02875 (2022).

4. DBeaver Community. *DBeaver Database Tool*. [https://dbeaver.io](https://dbeaver.io). Accessed: 2025-06-27.

5. Avrilia Floratou, Fotis Psallidas, Fuheng Zhao, Shaleen Deep, Gunther Hagleither, Wangda Tan, Joyce Cahoon, Rana Alotaibi, Jordan Henkel, Abhik Singla, Alex Van Grootel, Brandon Chow, Kai Deng, Katherine Lin, Marcos Campos, K. Venkatesh Emani, Vivek Pandit, Victor Shnayder, Wenjing Wang, and Carlo Curino. *NL2SQL is a solved problem... Not!*. In CIDR, 2024.

6. Hugging Face. *BLEU Metric*. [https://huggingface.co/spaces/evaluate-metric/bleu](https://huggingface.co/spaces/evaluate-metric/bleu). Accessed: 2025-06-27.

7. Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, and Nan Tang. *The Dawn of Natural Language to SQL: Are We Fully Ready?* Proceedings of the VLDB Endowment 17(11), 3318–3331 (2024). [PDF](https://www.vldb.org/pvldb/vol17/p3318-luo.pdf)

8. Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C.C. Chang, Fei Huang, Reynold Cheng, and Yongbin Li. *Can LLM already serve as a database interface? A big bench for large-scale database grounded text-to-SQLs*. In NeurIPS, 2024.

9. Michele Orrù. *Radon: Python tool for code metrics*. [https://radon.readthedocs.io/en/latest/intro.html](https://radon.readthedocs.io/en/latest/intro.html). Accessed: 2025-06-27.

10. Papers with Code. *WikiTableQuestions Dataset*. [https://paperswithcode.com/dataset/wikitablequestions](https://paperswithcode.com/dataset/wikitablequestions). Accessed: 2025-06-27.

11. Simone Papicchio, Paolo Papotti, and Luca Cagliero. *QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data*. In Conference on Neural Information Processing Systems (NeurIPS), 2023.

12. Rebecca J. Passonneau, Clay Morrison, Ashish Bhardwaj, David Stuart, and Ingo Scholtes. *Text-to-SQL in the Wild: A Naturally-Occurring Dataset Based on Stack Exchange Data*. arXiv preprint arXiv:2009.10297 (2020).

13. Cédric Renggli, Ihab F. Ilyas, and Theodoros Rekatsinas. *Fundamental Challenges in Evaluating Text2SQL Solutions and Detecting Their Limitations*. arXiv preprint arXiv:2501.18197 (2025). [DOI](https://doi.org/10.48550/ARXIV.2501.18197)

14. Immanuel Trummer. *How to Win a Prize: Analyzing SQL Query Optimization Competitions*. Proceedings of the VLDB Endowment 15(12), 2921–2933 (2022).

15. Yale LILY Lab. *Spider: A Large-Scale Human-Labeled Text-to-SQL Dataset*. [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider). Accessed: 2025-06-27.

16. Yibo Yao, Yanlin Wang, Yichong Cheng, Victor Zhong, Yining Zhang, Xi Victoria Lin, Xiao Du, Mu Li, Xiang Li, Zhouhan Li, Zhengbao Jiang, and Tao Yu. *ResdSQL: Decoupling Schema Linking and Reasoning for Text-to-SQL Parsers*. In EMNLP, 2023.

17. Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. In EMNLP, 2018.
