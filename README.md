## 双分支

本分支为生成GitHub pages所需要的Hexo环境，方便迁移

- 已设置default branch为hexo分支
- master分支：修改_config.yml中的deploy参数，分支应为master

## 将内容迁移到另一台电脑
大部分借助github分支，其中有一点是mathjax的格式问题，需要修改一下kramed渲染的escape和em两行代码，具体位置可查【参考资料2】
- escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
- em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,

## 参考资料
- https://www.jianshu.com/p/153490a029a5
- [将CSDN博客迁移至Hexo个人博客](https://baidinghub.github.io/2020/03/03/%E5%8D%9A%E5%AE%A2%E8%BF%81%E7%A7%BB/#%E5%B0%86CSDN%E5%8D%9A%E5%AE%A2%E8%BF%81%E7%A7%BB%E8%87%B3Hexo%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2)
