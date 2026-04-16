# Hello World Skill — Extended Examples

## Multi-language Greetings

**Chinese:**
```
你好，世界！欢迎来到 AgentWorld。
```

**Japanese:**
```
こんにちは、世界！AgentWorldへようこそ。
```

**Spanish:**
```
¡Hola, Mundo! Bienvenido a AgentWorld.
```

## Batch Greeting

Input: "Greet the team: Alice, Bob, Charlie"

Output:
```
Hey there, Alice! 👋
Hey there, Bob! 👋
Hey there, Charlie! 👋

Welcome to the team, everyone!
```

## Greeting with Context

Input: "It's the project launch day, greet everyone"

Output:
```
🚀 Happy Launch Day, Team!

Congratulations on reaching this milestone. Here's to a successful deployment!
```

## Error Handling Examples

**Empty name:**
```
Input:  --name "" --style casual
Output: Error: Name cannot be empty. Using default "World".
Result: Hey there, World! 👋
```

**Invalid style:**
```
Input:  --name "Alice" --style pirate
Output: Warning: Unknown style "pirate". Falling back to "casual".
Result: Hey there, Alice! 👋
```
