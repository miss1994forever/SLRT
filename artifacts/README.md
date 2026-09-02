# Local artifacts

此目录只定义本地外部资产的统一布局。除本说明外，内容均被 Git 忽略。

```text
artifacts/
└── checkpoints/
    └── online_slrt/
        └── cslr_best.ckpt
```

不要提交 checkpoint、下载凭据或带时效的私有链接。推荐通过复制或符号链接把服务器上的已有权重放到上述位置，使配置无需包含机器用户名和绝对路径。
