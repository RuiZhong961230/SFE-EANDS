# SFE-EANDS  
SFE-EANDS: A Simple, Fast, and Efficient Algorithm with External Archive and Normalized Distance-based Selection for High-Dimensional Feature Selection  

## Abstract
SFE (i.e., Simple, Fast, and Efficient) is a novel evolutionary approach to deal with the high-dimensional feature selection challenge, the excellent performance and easy implementation of SFE have quickly attracted widespread attention. However, the original SFE is prone to stagnation in the later stages of optimization and lacks effectiveness in escaping local optima since the single-agent search pattern. To handle these issues, we introduce two strategies (i.e., the External Archive and the Normalized Distance-based Selection mechanism) to improve the performance of SFE and propose SFE-EANDS. The external archive saves the optimal solution found so far to guide the direction of optimization, and the normalized distance-based selection offers a probability for an inferior solution to be accepted. To comprehensively evaluate the performance of SFE-EANDS, we implement experiments on 21 high-dimensional datasets, and ten advanced approaches are applied as competitor algorithms. The experimental results and statistical analyses confirm the effectiveness and efficiency of our proposed SFE-EANDS. Moreover, we have integrated four widely used classifiers into the SFE-EANDS framework to investigate its robustness and scalability. Based on numerical experiments, we recommend integrating the SFE-EANDS with the k-nearest neighbors (KNN) with $k$=1 to solve the high-dimensional feature selection tasks.  

## Citation
@article{Zhong:25,  
  title={SFE-EANDS: A Simple, Fast, and Efficient Algorithm with External Archive and Normalized Distance-based Selection for High-Dimensional Feature Selection},  
  author={Rui Zhong and Yang Cao and Essam H. Houssein and Jun Yu and Masaharu Munetomo},  
  journal={Cluster Computing},  
  pages={1-24},  
  year={2025},  
  volume={28},  
  publisher={Springer}  
}

## Datasets and Libraries
The datasets can be downloaded from http://csse.szu.edu.cn/staff/zhuzx/datasets.html, https://jundongl.github.io/scikitfeature/datasets.html, https://figshare.com/articles/dataset/Microarray_data_rar/7345880/2, and https://file.biolab.si/biolab/supp/bicancer/projections/info/gastricGSE2685.html.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
