#!/usr/bin/env python3
"""
Fix all except blocks that have wrong indentation.
The pattern is: except blocks with 8 spaces should have 12 spaces to match their try blocks.
"""

with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix pattern: "        except Exception as e:\n            print" -> "            except Exception as e:\n                print"
import re

# Pattern 1: 8 spaces except -> 12 spaces except (and fix print line too)
content = re.sub(
    r'\n        except Exception as e:\n            print\(f"   ⚠️',
    r'\n            except Exception as e:\n                print(f"   ⚠️',
    content
)

# Pattern 2: 12 spaces except -> 16 spaces except (nested deeper)
content = re.sub(
    r'\n            except Exception as e:\n                print\(f"   ⚠️',
    r'\n                except Exception as e:\n                    print(f"   ⚠️',
    content
)

# Pattern 3: 16 spaces except -> 20 spaces except (even deeper)
content = re.sub(
    r'\n                except Exception as e:\n                    print\(f"   ⚠️',
    r'\n                    except Exception as e:\n                        print(f"   ⚠️',
    content
)

# Pattern 4: 20 spaces except -> 24 spaces except
content = re.sub(
    r'\n                    except Exception as e:\n                        print\(f"   ⚠️',
    r'\n                        except Exception as e:\n                            print(f"   ⚠️',
    content
)

with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed except block indentation')
