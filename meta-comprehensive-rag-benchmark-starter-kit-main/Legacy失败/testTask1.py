import os
import sys
import types

from PIL import Image

# 从 .env 或系统环境变量读取；本地请在项目根目录创建 .env 文件。
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 本地 smoke test 可能尚未安装 cragmm-search-pipeline。
# 这里仅为导入 BaseAgent 提供最小 stub；真实评测会使用官方 cragmm_search 包。
if "cragmm_search.search" not in sys.modules:
    cragmm_search = types.ModuleType("cragmm_search")
    search_module = types.ModuleType("cragmm_search.search")

    class UnifiedSearchPipeline:
        pass

    search_module.UnifiedSearchPipeline = UnifiedSearchPipeline
    cragmm_search.search = search_module
    sys.modules["cragmm_search"] = cragmm_search
    sys.modules["cragmm_search.search"] = search_module

from agents.Task1KGAgent import Task1KGAgent


class FakeImageSearchPipeline:
    """用于本地 smoke test 的假图像检索 API，不下载 CRAG-MM 索引。"""

    def __call__(self, image, k=5):
        return [
            {
                "score": 0.91,
                "url": "https://example.com/building.jpg",
                "entities": [
                    {
                        "entity_name": "8 Spruce Street",
                        "entity_attributes": {
                            "name": "8 Spruce Street<br />(New York by Gehry)",
                            "architect": "[[Frank Gehry]]",
                            "floor_count": "76",
                            "opening": "February 2011",
                            "address": "8 Spruce Street, [[Manhattan]], New York",
                        },
                    }
                ],
            }
        ][:k]


def main():
    agent = Task1KGAgent(search_pipeline=FakeImageSearchPipeline())
    image = Image.new("RGB", (32, 32), color="white")
    queries = [
        "Who is the architect of this building?",
        "How many floors does it have?",
        "What is this building called?",
    ]
    answers = agent.batch_generate_response(queries, [image] * len(queries), [[] for _ in queries])
    for query, answer in zip(queries, answers):
        print(f"Q: {query}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
