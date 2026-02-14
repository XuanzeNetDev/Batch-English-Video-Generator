# Git 工作流程

## 分支说明

- **master** - 完整代码（包含核心生成器）
- **public-demo** - 公开版本（不包含核心代码）

## 远程仓库

- **public** - 公开仓库：https://github.com/XuanzeNetDev/Batch-English-Video-Generator
- **private** - 私有仓库：https://github.com/XuanzeNetDev/Batch-English-Video-Generator-Full

## 更新流程

### 1. 更新文档或Demo文件

```bash
# 在 master 分支修改
git checkout master
# 修改文件...
git add .
git commit -m "更新说明"

# 推送到私有仓库
git push private master

# 切换到公开分支并合并更改
git checkout public-demo
git merge master
git push public public-demo:master
```

### 2. 更新核心代码

```bash
# 在 master 分支修改
git checkout master
# 修改核心代码...
git add .
git commit -m "更新核心代码"

# 只推送到私有仓库
git push private master

# 不要推送到公开仓库！
```

## 注意事项

- ⚠️ 核心代码文件永远不要推送到公开仓库
- ⚠️ public-demo 分支已经删除了核心代码，合并时会保持删除状态
- ✅ 文档更新需要同时推送到两个仓库
