파이썬 함수의 인자로 다양한 인자들이 존재한다.

- 기본 인자
    - 위치 인수
    - 키워드 인수

- 가변 인자
    - 위치 가변 인자 (*args)
    - 키워드 가변 인자 (**kwargs)

### 0. 기본 인자 : 위치 인자, 키워드 인자

간단히 설명하면 위치 인자는 위치 자체로 함수의 인자로 들어가고 <br>
키워드 인자는 키워드 (ex. y = 3)를 통해 함수의 인자로 들어간다


```python
def func0(x, y):
    print(f"x : {x}, y : {y}")
    return x + y

func0(10, y = 3)
```

    x : 10, y : 3





    13



가변 인자 : 함수가 임의의 개수의 인수를 받을 수 있도록 하는 기능<br>
기본 인자와 비교하면 기본인자는 1:1 매핑, 가변 인자는 1:N 매핑이 된다고 이해하면 된다.

### 1. 위치 가변 인자 : *args

전달된 모든 위치 인수들은 **튜플**로 묶여서 함수 내에서 사용 <br>
위치 인자와 동일하게 위치에 따라 받는 인자들이 정해진다


```python
def func1(x, *args):
    print(f"x : {x}")
    print(f"*args : {args}")

func1(1, [10, 5], 3)
```

    x : 1
    *args : ([10, 5], 3)


이때 *args의 위치에 따라 아래와 같은 오류가 발생 할 수 있다.


```python
def func2(x, *args, y):
    print(f"x : {x}")
    print(f"*args : {args}")
    print(f"y : {y}")

func2(1, 10, 5, 3)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[22], line 6
          3     print(f"*args : {args}")
          4     print(f"y : {y}")
    ----> 6 func2(1, 10, 5, 3)


    TypeError: func2() missing 1 required keyword-only argument: 'y'


위와 같은 경우 가변 위치 인자인 *args 때문에 함수에 y에 해당하는 인자가 전달되지 않은 것이다. <br>
따라서 키워드 인자를 통해 다음과 같이 y에 값을 넣어 주어야 한다.


```python
func2(1, 10, 5, y=3)
```

    x : 1
    *args : (10, 5)
    y : 3


### 2. 키워드 가변 인자 : **kwargs

키워드 가변 인자는 함수가 임의의 개수의 키워드 인자를 받을 수 있도록 한다.

**kwargs를 사용하면, 전달된 모든 키워드 인수들은 **딕셔너리**로 묶여서 함수 내에서 사용한다.


```python
def func3(x, **kwargs):
    print(f"x : {x}")
    for (key, value) in kwargs.items():
        print(f" key & value : ({key}, {value})")

func3(1, a = 10, c = 5, d = 3)
```

    x : 1
     key & value : (a, 10)
     key & value : (c, 5)
     key & value : (d, 3)


**kwargs는 딥러닝 모델을 정의할 때 다양한 인자들 (예를 들어 scale, aspect ratio 등등)을 **kwargs에 담아서 (key, value) 쌍으로 꺼내서 사용하게 된다.

이때 (불필요한 생각이지만) 키워드 가변 인자 뒤에 키워드 인자를 두면 어떻게 되는지 궁금할 수 있다. <br>

아래의 코드에서 보면 오류가 발생하는데, 이는 키워드 가변인자는 함수의 정의에서 위치 인수와 키워드 전용 인수 뒤에 와야하는 파이썬 문법의 조건이 있다고 한다.


```python
def func4(x, **kwargs, y):
    print(f"x : {x}")
    print(f"**kwargs : {kwargs}")
    print(f"y : {y}")

func4(1, a = 10, c = 5, d = 3, y = 1)
```


      Cell In[29], line 1
        def func4(x, **kwargs, y):
                               ^
    SyntaxError: invalid syntax



### 3. 위치 가변 인자와 키워드 가변 인자 함께 사용

앞에서 구현한 함수들과 여러가지 오류들(제약사항)을 생각하며 예시 함수를 구현해보자


```python
def func5(x, *args, **kwargs):
    print(f"x : {x}")
    print(f"*args : {args}")
    for (key, value) in kwargs.items():
        print(f" key & value : ({key}, {value})")

func5(10, "위치 인자", 1, 2, ["hi", "hello"], a="키워드", b="가변", c="인자")
```

    x : 10
    *args : ('위치 인자', 1, 2, ['hi', 'hello'])
     key & value : (a, 키워드)
     key & value : (b, 가변)
     key & value : (c, 인자)


### 4. 마지막으로 DL 모델 구현에서 사용되는 예시를 보며 마무리 하겠다.

아래는 SwinTransformer에 대한 구현 내용이다. <br>
출처 (https://github.com/berniwal/swin-transformer-pytorch) <br>
직접 코드를 실행하진 않으며 인자로 전달되는 kwargs를 위주로 보길 바란다.


```python
net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)
```


```python
def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
```

hidden_dim, layers, heads 외의 인자들은 모두 kwargs로 전달된다.

이때 swin_t 함수 선언의 매개변수 **kwargs는 위의 설명대로 **가변 키워드 인자**를 의미한다. <br>
그리고 SwinTransformer의 인자로 전달되는 **kwargs는 **딕셔너리 언패킹**을 의미한다.


```python
class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
```

굳이 **kwargs를 쓰지 않고 모든 키워드 인자를 입력해서 쓰면 어떨까라는 생각도 하지만 <br>
DL 모델들의 특성상 바꿔줘야할 하이퍼파라미터가 무수히 많기 때문에 이러한 가변 인자를 통해서 코드를 좀 더 간결하게 만드는게 아닐까 싶다.
