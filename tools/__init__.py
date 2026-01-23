"""
tools 包 - 辅助工具模块

提供向量生成等工具
"""

from .generate_dense_embedding import (
    ModelArgs as DenseModelArgs,
    get_model as get_dense_model,
    generate_dense,
    save_result as save_dense_result,
    main as generate_dense_main
)

# from .generate_sparse_embedding import (
#     ModelArgs as SparseModelArgs,
#     get_model as get_sparse_model,
#     generate_sparse,
#     save_result as save_sparse_result,
#     main as generate_sparse_main
# )

# from .view_es_documents import (
#     ESViewer,
#     main as view_es_main
# )

__all__ = [
    # 密集向量生成
    'DenseModelArgs',
    'get_dense_model',
    'generate_dense',
    'save_dense_result',
    'generate_dense_main',
    
    # # 稀疏向量生成
    # 'SparseModelArgs',
    # 'get_sparse_model',
    # 'generate_sparse',
    # 'save_sparse_result',
    # 'generate_sparse_main',
    
    # # ES查看工具
    # 'ESViewer',
    # 'view_es_main',
]
