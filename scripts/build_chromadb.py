import sys
import os
import json
import argparse

# --- 解析参数 ---
def get_args():
    parser = argparse.ArgumentParser(description="Building a ChromaDB vector database for legal documents")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing processed legal documents (each law/ has articles/ under it)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for saving ChromaDB databases"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="law_articles",
        help="ChromaDB collection name (default: law_articles)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the database directory already exists, should it be overwritten and recreated (default: False)"
    )
    return parser.parse_args()

# --- 设置路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- 依赖 ---
from src.utils.parse_law_json import parse_law_json_to_docs
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    args = get_args()

    embedding = HuggingFaceEmbeddings(
        model_name="ai-forever/ru-en-RoSBERTa",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    all_enhanced_documents = []

    for law in os.listdir(args.input_dir):
        article_dir = os.path.join(args.input_dir, f"{law}/articles")
        if not os.path.isdir(article_dir):
            continue
        for file in os.listdir(article_dir):
            with open(os.path.join(article_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            all_enhanced_documents.extend(parse_law_json_to_docs(data))

    # 初始化数据库
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=args.output_dir
    )

    # 检查 collection 是否存在
    existing_collections = [c.name for c in vectorstore._client.list_collections()]
    if args.collection_name in existing_collections:
        if args.overwrite:
            print(f"⚠️ Collection {args.collection_name} already exists, deleting and rebuilding...")
            vectorstore._client.delete_collection(name=args.collection_name)
        else:
            print(f"❌ Collection {args.collection_name} already exists. Use --overwrite to rebuild it.")
            sys.exit(1)

    # 创建 collection 并写入数据
    new_collection = Chroma.from_documents(
        documents=all_enhanced_documents,
        embedding=embedding,
        persist_directory=args.output_dir,
        collection_name=args.collection_name
    )

    print(f"✅ Collection '{args.collection_name}' created successfully in database {args.output_dir}")

if __name__ == "__main__":
    main()
