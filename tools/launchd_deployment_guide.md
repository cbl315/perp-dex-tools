# macOS launchd 重启方案部署指南

## 概述

本方案使用 macOS 原生的 launchd 服务管理器来实现交易机器人的智能重启机制。launchd 只在技术异常（退出码1）时自动重启，在策略性退出（止损、停止价格、仓位不匹配）时不重启。

## 退出码策略

| 退出码 | 原因 | 重启行为 | 说明 |
|--------|------|----------|------|
| **0** | 止损触发、停止价格、仓位不匹配 | **不重启** | 策略性退出，需要人工干预 |
| **1** | 网络错误、技术异常 | **自动重启** | 技术故障，自动恢复 |

## 部署步骤

### 1. 创建日志目录
```bash
mkdir -p /Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/logs
```

### 2. 复制 plist 文件到系统目录
```bash
# 复制到用户级服务目录（推荐）
cp com.yourquantguy.trading-bot-lighter-eth.plist ~/Library/LaunchAgents/

# 或者复制到系统级服务目录（需要管理员权限）
# sudo cp com.yourquantguy.trading-bot-lighter-eth.plist /Library/LaunchDaemons/
```

### 3. 加载并启动服务
```bash
# 加载服务
launchctl load ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist

# 启动服务
launchctl start com.yourquantguy.trading-bot-lighter-eth
```

### 4. 验证服务状态
```bash
# 查看服务状态
launchctl list | grep com.yourquantguy.trading-bot-lighter-eth

# 查看服务详情
launchctl print gui/$(id -u)/com.yourquantguy.trading-bot-lighter-eth
```

## 管理命令

### 启动服务
```bash
launchctl start com.yourquantguy.trading-bot-lighter-eth
```

### 停止服务
```bash
launchctl stop com.yourquantguy.trading-bot-lighter-eth
```

### 重启服务
```bash
launchctl stop com.yourquantguy.trading-bot-lighter-eth
launchctl start com.yourquantguy.trading-bot-lighter-eth
```

### 卸载服务
```bash
launchctl unload ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist
```

### 重新加载配置
```bash
# 先卸载再加载
launchctl unload ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist
launchctl load ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist
```

## 监控和日志

### 查看服务日志
```bash
# 查看标准输出日志
tail -f /Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/logs/trading-bot.log

# 查看错误日志
tail -f /Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/logs/trading-bot-error.log
```

### 查看系统日志
```bash
# 查看 launchd 相关日志
log show --predicate 'subsystem == "com.apple.xpc.launchd"' --info --last 1h

# 查看特定服务的日志
log show --predicate 'process == "uv"' --info --last 1h
```

### 实时监控
```bash
# 监控服务状态
while true; do
    launchctl list | grep com.yourquantguy.trading-bot-lighter-eth
    sleep 10
done
```

## 智能重启行为

### 自动重启的情况（退出码1）
- **网络连接错误**
- **API调用失败**
- **其他技术异常**
- **程序内部错误**

### 不重启的情况（退出码0）
- **止损触发** - 保护资金安全
- **停止价格触发** - 避免不利行情
- **仓位不匹配** - 需要人工干预
- **用户手动停止** - 用户主动控制

## 故障排除

### 服务无法启动
```bash
# 检查 plist 文件语法
plutil -lint ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist

# 检查文件权限
ls -la ~/Library/LaunchAgents/com.yourquantguy.trading-bot-lighter-eth.plist

# 检查工作目录和可执行文件
ls -la /Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/runbot.py
which uv
```

### 服务频繁重启
```bash
# 查看重启历史
log show --predicate 'subsystem == "com.apple.xpc.launchd"' --info --last 30m | grep "com.yourquantguy.trading-bot-lighter-eth"

# 检查退出码
tail -n 20 /Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/logs/trading-bot-error.log
```

### 手动测试重启机制
```bash
# 模拟技术异常（应该重启）
launchctl stop com.yourquantguy.trading-bot-lighter-eth
# 等待10秒观察是否自动重启

# 模拟策略退出（不应该重启）
# 在代码中触发止损条件，观察是否停止重启
```

## 开机自启

服务配置了 `RunAtLoad true`，系统重启后会自动启动交易机器人。

## 配置说明

### 关键配置项
- **Label**: 服务唯一标识符
- **ProgramArguments**: 执行命令和参数
- **WorkingDirectory**: 工作目录
- **StandardOutPath/StandardErrorPath**: 日志文件路径
- **KeepAlive/ExitCodes**: 智能重启配置
- **RunAtLoad**: 开机自启
- **StartInterval/ThrottleInterval**: 重启间隔控制

### 自定义配置
如果需要修改交易参数，编辑 plist 文件中的 `ProgramArguments` 部分：
```xml
<array>
    <string>/usr/local/bin/uv</string>
    <string>run</string>
    <string>python</string>
    <string>/Users/caibiliang/code/github.com/your-quantguy/perp-dex-tools/runbot.py</string>
    <string>--exchange</string>
    <string>lighter</string>
    <string>--ticker</string>
    <string>ETH</string>
    <string>--quantity</string>
    <string>0.1</string>
    <!-- 添加其他参数 -->
</array>
```

## 优势

1. **macOS 原生**: 无需安装额外软件
2. **智能重启**: 只在技术异常时重启
3. **稳定可靠**: 系统级服务管理
4. **开机自启**: 系统重启后自动启动
5. **完善日志**: 支持详细的日志记录
6. **易于管理**: 标准的 launchctl 命令

这个方案提供了完整的智能重启机制，确保交易机器人在技术故障时自动恢复，同时在策略性退出时停止运行，保护资金安全。
