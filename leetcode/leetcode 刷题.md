# leetcode 刷题

## 数组

### 二分查找

https://leetcode-cn.com/problems/binary-search/

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        list_length = len(nums)
        left = 0
        right = list_length - 1 
        mid_index = ( left + right ) // 2
        while left <= right:
            if target == nums[mid_index]:
                return mid_index
            elif target < nums[mid_index]:
                # 在左边
                right = mid_index - 1
                mid_index = (left + right) // 2
                continue
            else:
                # 在右边
                left = mid_index + 1
                mid_index = (left + right) // 2
                continue
        return -1
```

使用二分查找的前提条件是**有序排列的数组**。

设置一个left，right和mid

其中mid = (left + right) // 2，即在[left, right]区间的中间

#### 第一种

循环结束条件是left > right，如果mid是要的答案就返回，否则修改left 或者right 和mid

在mid左边则修改right和mid，right修改为mid-1；在mid右边则修改left和mid，left修改为mid + 1；

#### 第二种

循环结束条件是left >= right，如果mid是要的答案就返回，否则修改left 或者right 和mid

在mid左边则修改right和mid，right修改为mid；在mid右边则修改left和mid，left修改为mid + 1；



#### 34 [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

思路：使用两次二分查找，分别寻找元素的第一个位置和最后一个位置。

即寻找第一个 >= target 和最后一个 <= target

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 因为是排序数组，所以使用二分查找
        nums_len = len(nums)
        left = 0
        right = nums_len - 1
        mid = (right + left) // 2
        [left_ans, right_ans] = [-1, -1]

        # 查找第一个 >= target 的，也就是left_ans
        while left < right:
            if nums[mid] >= target:
                right = mid
                mid = ( left + right) // 2
            else:
                left = mid + 1
                mid = ( left + right ) // 2

        if mid < 0:
            return [-1, -1]        
        if nums[mid] != target:
            return [left_ans, right_ans]

        left_ans = mid

        left = 0
        right = nums_len - 1
        mid = (right + left) // 2
        # 查找最后一个 <= target的，也就是right_ans
        while left < right:
            if nums[mid] <= target:
                left = mid
                mid = (left + right + 1) // 2
            else:
                right = mid - 1
                mid = (left + right + 1) // 2
                
        right_ans = mid
        return [left_ans, right_ans]
        


```

想要使得最后的mid就是我们要找的值，需要选择没有left=right的形式，如果要让mid最后和left相等，那么每次mid要向上取整，即需要(left+right+1) // 2；同理，如果需要最后mid和right相等，那么每次mid需要向下取整，



### 移除元素

要求:空间复杂度是O（1）

#### 第一种：暴力解法

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        count = 0
        i = 0
        nums_len = len(nums)
        while i < nums_len:
            if nums[i] == val:
                for j in range(i, nums_len - 1):
                    nums[j] = nums[j + 1]
                count = count + 1
                i = i - 1
                nums_len = nums_len - 1
            i = i+1
        return len(nums) - count
```

嵌套两层循环，一层循环遍历数组，一层循环将后面的元素前移。

需要注意的是前移之后原本在i+1的元素就放到i上了。

最后返回的是len - count~

#### 第二种 双指针法

通过一个快指针和一个慢指针在一个for循环下完成两个for循环的工作。

遇到不需要的值就停下，用下一个正常的值覆盖掉不需要的值即可。

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        nums_len = len(nums)
        quickIndex = 0
        slowIndex = 0
        while quickIndex < nums_len:
            if nums[quickIndex] != val:
                nums[slowIndex] = nums[quickIndex]
                slowIndex = slowIndex + 1
            quickIndex = quickIndex + 1
        return slowIndex
```

### 长度最小的子数组

#### 暴力遍历

#### 滑动窗口

**要让l指针和r指针都只移动一次，这样滑动窗口才有意义，时间复杂度才可以降到O(n)**

```typescript
function minSubArrayLen(target: number, nums: number[]): number {
    let min_len = nums.length + 1;
    let low_p = 0;
    let high_p = 0;
    let temp_sum = 0;
    let temp_len = 0;
    while (high_p < nums.length) {
        temp_sum += nums[high_p];
        temp_len ++;
        while (temp_sum >= target) {
            min_len = min_len < temp_len ? min_len : temp_len;
            temp_sum = temp_sum - nums[low_p];
            low_p ++;
            temp_len --;
        }
        high_p ++
    }

    if(min_len > nums.length) {
        return 0
    }else {
        return min_len;
    }
};
```

其实也是双指针的一种变形，所以其实每一个元素都是被使用了两次，一次high_p，一次low_p，所以时间复杂度是O（n）

* 窗口内是满足和>=s的连续子数组
* 窗口的起始位置如何移动：如果窗口值大于s，则需要向前移动了。
* 结束位置如何移动：遍历，所以就是往前走。

![leetcode_209](https://img-blog.csdnimg.cn/20210312160441942.png)



[904.水果成篮](https://leetcode-cn.com/problems/fruit-into-baskets/)

```typescript
function totalFruit(fruits: number[]): number {
    // 连续区间的问题，简单的方法需要两重遍历，所以使用双指针
    let l = 0;
    let maxLen = 0;
    let last_index = 0;
    let packet = [fruits[0]];

    for (let r=0; r < fruits.length; r ++) {
        if (!packet.includes(fruits[r])) {
            if (packet.length == 1) {
                packet[1] = fruits[r];
            } else {
                // 遇到了第三种水果
                l = last_index;
                packet[0] = fruits[r-1];
                packet[1] = fruits[r]
            }
        }
        if(fruits[r] != fruits[last_index]) {
            last_index = r;
        }
        maxLen = Math.max(maxLen, r-l+1)
    }
    return maxLen;
};
```

其实也是求连续区间中的最值。

* 遇到第三种水果的时候需要移动 l 指针，并将l指针移动到前一种水果的位置。
* 前一种水果显然是fruit[r-1]
* 难点在于获得last_index，也就是上一种水果连续的最早的index，每次判断现在保存的fruit[last_index]和fruit[r]是否一样，一样就不改变last_index，否则修改为r。

### 螺旋矩阵

本质就是模拟过程。

**注意**： 循环不变量原则

例如没一次循环都是左闭右开，这样不断循环

```typescript
function generateMatrix(n: number): number[][] {
    let matrix:number[][] = Array.from({length: n}).map(() => new Array(n));
    let loop_num:number = (n+1) >> 1;
    let count:number = 0;
    let number_in_matrix:number = 1;
    while(loop_num --) {
        // 左闭右开
        let len_now:number = n - 2*count;
        console.log(len_now + '\n');
        if(len_now == 1) {
            matrix[Math.floor(n/2)][Math.floor(n/2)] = number_in_matrix;
            break;
        }

        for(let i:number = 0; i < len_now-1; i++) {
            matrix[count][i+count] = number_in_matrix++; 
        }

        for(let j:number = 0; j < len_now-1; j++ ) {
            matrix[j+count][n-1-count] = number_in_matrix ++;
        }

        for(let i:number = 0; i < len_now-1; i++) {
            matrix[n-1-count][n-1-i-count] = number_in_matrix ++;
        }

        for(let j:number = 0; j<len_now-1; j++) {
            matrix[n-1-j-count][count] = number_in_matrix ++;
        }
        
        count ++;
    }

    return matrix
};
```

对于ts/js，初始化矩阵的问题

```typescript
let matrix = Array.from({length:n}).map(() => new Array(n))
// 如果使用fill函数则都是同一个数组地址，就不行
// new Array(n).fill(new Array(n))
```



## 链表

### 反转链表

https://leetcode-cn.com/problems/reverse-linked-list/

思路：首先再建立一个新的链表是浪费，所以肯定是在原先的链表上进行操作。设置一个pre指针和一个cur指针，然后不断指向前面即可。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = ListNode()
        cur = head

        while cur != None:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        
        head = pre
        return head
            
```



### 两两交换链表中的节点

https://leetcode-cn.com/problems/swap-nodes-in-pairs/

思路：模拟的思想

要注意的就是循环的停止条件

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head == None or head.next == None: 
            return head
        

        p = ListNode()
        q = ListNode()
        realHead = head.next
        p = head
        q = head.next
        tmp = ListNode()
        while p != None and q!= None:
            p.next = q.next
            q.next = p
            if p != head:
                tmp.next = q
            tmp = p
            p = p.next
            if p != None:
                q = p.next
        return realHead
```



### 删除链表的倒数第N个节点

思路：

1. 直接遍历的思路，等于两次循环
2. **双指针** 先让后面的指针移动n个，那么再一起移动到最后，则就是倒数第n个位置，因为已经让这两个指针之间空出了n个位置。 注意可以使用虚拟头节点会简单，**最终返回的是虚拟头节点的后一个**！



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # 双指针 一次遍历
        virtualHead = ListNode(0, head)
        p = virtualHead
        q = virtualHead
        count = 0
        while q.next != None: 
            q = q.next
            count = count + 1
            if count >= n+1:
                p = p.next
        
        
        p.next = p.next.next
        return virtualHead.next
```



### 面试题 02.07. 链表相交

思路：

1. 最简单的就是暴力循环，但是复杂度过大，超时了
2. 思考一下本质问的是啥，其实前面长出的部分都不可能相同，后面也之后一样长对应的部分可以相同，所以求两个链表的长度，后将指针移动到同等的位置，开始比较，则复杂度会是O（n）,只有三次一次循环

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 求两个链表的长度，后将指针移动到同等的位置，开始比较，则复杂度会是O（n）,只有三次一次循环
        curA = headA
        [countA,countB] = [0, 0]
        curB = headB
        while curA != None:
            curA = curA.next
            countA = countA + 1
        while curB != None:
            curB = curB.next
            countB = countB + 1

        curA = headA
        curB = headB
        countMin = min(countA, countB)
        countMax = max(countA, countB)
        count = countMax - countMin

        for index in range(0, count):
            if countB > countA:
                curB = curB.next
            else:
                curA = curA.next

        while curA != None and curB != None:
            if curA == curB:
                return curA
            curA = curA.next
            curB = curB.next
        
        return None

```


### 环形链表II

https://leetcode-cn.com/problems/linked-list-cycle-ii/

思路：

* 如何确定有环？

  * 用两个快慢指针，一定会相遇，一个一次走两步，一个一次走一步

* 如何确定入口位置？

  * 假设从头结点到环形入口节点 的节点数为x。 环形入口节点到 fast指针与slow指针相遇节点 节点数为y。 从相遇节点 再到环形入口节点节点数为 z。 如图所示：

  * ![image-20211110205227184](C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20211110205227184.png)

  * 那么相遇时： slow指针走过的节点数为: `x + y`， fast指针走过的节点数：`x + y + n (y + z)`，n为fast指针在环内走了n圈才遇到slow指针， （y+z）为 一圈内节点的个数A。

    因为fast指针是一步走两个节点，slow指针一步走一个节点， 所以 fast指针走过的节点数 = slow指针走过的节点数 * 2：

    ```
    (x + y) * 2 = x + y + n (y + z)
    ```

    两边消掉一个（x+y）: `x + y = n (y + z)`

    因为要找环形的入口，那么要求的是x，因为x表示 头结点到 环形入口节点的的距离。

    所以要求x ，将x单独放在左面：`x = n (y + z) - y` ,

    再从n(y+z)中提出一个 （y+z）来，整理公式之后为如下公式：`x = (n - 1) (y + z) + z` 注意这里n一定是大于等于1的，因为 fast指针至少要多走一圈才能相遇slow指针。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # 如果相遇
            if slow == fast:
                p = head
                q = slow
                while p != q:
                    p = p.next
                    q = q.next
                return p

        return None
```





### 两个数组的交集

用python非常简单，利用set数据类型

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1, nums2 = set(nums1), set(nums2)
        return list(nums1.intersection(nums2))

```



### 快乐数

难点在于如何判断是**无限循环**

**无限循环**意味着数字会重复出现，就可以使用集合。如果重复了则说明无限循环了

如何获得每一位的数字看calculate_happy(num)

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        n_str = str(n)
        one_set = set()
        temp = 0
        for i in n_str:
            temp += int(i) ** 2
        
        temp_set = set([temp])
        while ((one_set.union(temp_set) != one_set) and temp != 1):
            one_set = one_set.union(temp_set)
            n_str = str(temp)
            temp = 0
            for i in n_str:
                temp += int(i) ** 2
            temp_set = set([temp])

        if temp == 1:
            return True
        else:
            return False
```

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        def calculate_happy(num):
            sum_ = 0
            while num:
                sum_ += (num % 10) ** 2
                num = num // 10
            return sum_
        
        record = set()

        while True:
            n = calculate_happy(n)
            if n == 1:
                return True
            
            if n in record:
                return False
            else:
                record.add(n)
            
```



### 两数之和

1. 暴力解法	
   * 简单但有效

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, num1 in enumerate(nums):
            for j, num2 in enumerate(nums[i+1:]):
                if num1 + num2 == target:
                    return [i,  i+j+1]

```

2. 使用字典，理由：

   * 需要保存数和下表，key value
   * 字典的查询效率最差也是O(logn)
   * 使用值作为索引，index就变成value了
   * 如果已经有满足和的那么直接返回即可。

   ```python
   class Solution:
       def twoSum(self, nums: List[int], target: int) -> List[int]:
           new_dict = dict()
   
           for index, num in enumerate(nums):
               if target - num not in new_dict:
                   new_dict[num] = index
               else:
                   return [new_dict[target-num], index]
   ```

3. 双指针

   * 对于双重循环可以尝试
   * 先排序时间复杂度O（nlogn）
   * 先画图推导，一边从最小的出发，一边从最大的出发，逐渐往中间靠拢。

   ```python
   class Solution:
       def twoSum(self, nums: List[int], target: int) -> List[int]:
           nums_s = sorted(nums)
           left = 0
           right = len(nums) - 1 
           ans_num = [-1, -1]
           stop_right = False
           while left < right:
               if nums_s[left] + nums_s[right] == target:
                   ans_num[0] = nums_s[left]
                   ans_num[1] = nums_s[right]
                   break
               elif nums_s[left] + nums_s[right] > target:
                   right -= 1
               else:
                   left += 1
           
           return_value = [0, 0]
           flag1 = True
           flag2 = True
           # print(ans_num)
           for i, value in enumerate(nums):
               if value == ans_num[0] and flag1:
                   # print(value, ans_num[0])
                   return_value[0] = i
                   flag1 = False
                   continue
               if value == ans_num[1] and flag2:
                   # print(value, ans_num[1])
                   return_value[1] = i
                   flag2 = False
               
               if flag2 == False and flag1 == False:
                   break
           return return_value
   ```

   

###  四数相加II

* hash map

* 通过hash的**key来保存值**， value根据需要去保存，比如保存有多少个这个key的数字

* 先判断前两个之和，保存该和有多少个

* 后两个求和的相反数和已经保存的key一样，则加上这个key对应的value则是可以生成的组数

* ```python
  class Solution:
      def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
          map_1 = dict()
          count = 0
          for a in nums1:
              for b in nums2:
                  if a + b in map_1:
                      map_1[a+b] += 1
                  else:
                      map_1[a+b] = 1
          
          for c in nums3:
              for d in nums4:
                  key = (-c-d)
                  if key in map_1:
                      count += map_1[key]
  
          return count             
  ```

  

### 赎金信

* 简单的hash

* 使用key保存字符，value保存出现的次数

* ```python
  class Solution:
      def canConstruct(self, ransomNote: str, magazine: str) -> bool:
          map_1 = dict()
          for i, c in enumerate(ransomNote):
              if c not in map_1:
                  map_1[c] = 1
              else:
                  map_1[c] += 1
          
          for i, c in enumerate(magazine):
              if (c in map_1) and (map_1[c] != 0):
                  map_1[c] -= 1
          
          for i, c in enumerate(ransomNote):
              if map_1[c] != 0:
                  return False
          
          return True
  ```





### 三数之和

* 可以把问题转换成两数字之和为另一个数的相反数的问题
* 双指针
* 难点在于重复去除
  * 输出的数组不能有重复的
    * 因此遇到重复的数组需要跳过，左边相等与右边相等都需要判断
    * 两边都不想等的时候，肯定是左右都要变的，不然不可能会继续保持0

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 双指针法, 先选择一个数，然后就变成了双数之和
        return_list = []
        # nums = list(set(m))
        nums = sorted(nums)
        for i in range(0, len(nums)):
            target = (-nums[i])
            left, right = i+1, len(nums) - 1
            if i >= 1 and nums[i] == nums[i - 1]:
                continue
            if nums[i] > 0:
                break
            while left < right:
                if nums[left] + nums[right] == target:
                    return_list.append([-target, nums[left], nums[right]])
                    while left != right and nums[left] == nums[left + 1]: left += 1
                    while left != right and nums[right] == nums[right - 1]:right -= 1
                    left += 1
                    right -= 1
                elif nums[left] + nums[right] > target:
                    right -= 1
                else:
                    left += 1

        
        return return_list

```



### 四数之和

* 和三数之和类似
* 现在几数之和都会了
* 判断边界，就是如果这个和上一个相等，那么就continue
  * 因为如果相等，那么想要成为同一个target的值也是一样的，但是范围变小了，所以肯定是子集

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums = sorted(nums)
        return_list = []
        for i in range(0, len(nums)):
            if i>0 and nums[i] == nums[i-1]: continue
            for j in range(i + 1, len(nums)):
                left, right = j + 1, len(nums) - 1
                if j > i + 1 and nums[j] == nums[j-1]:continue
                if left >= len(nums) - 1:
                    break
                while left < right:
                    total = nums[i] + nums[j] + nums[left] + nums[right]
                    if total > target:
                        right -= 1
                    elif total < target:
                        left += 1
                    else:
                        return_list.append([nums[i], nums[j], nums[left], nums[right]])
                        while left != right and nums[left] == nums[left + 1] : left = left + 1
                        while left != right and nums[right] == nums[right - 1] : right = right - 1
                        left += 1
                        right -= 1
            
        return return_list

```



### 实现strStr()

* KMP

* 两个部分，一个求next数组，一个利用next数组进行匹配

  * ```python
    
    def getNext(self, str_):
            j = -1
            next_arr = [-1]
            for i in range(1, len(str_)):
                while j >= 0 and str_[i] != str_[j+1]:
                    j = next_arr[j]
                if str_[i] == str_[j+1]:
                    j += 1
                next_arr.append(j)
            return next_arr
    ```

  * ```python
    def strStr(self, haystack: str, needle: str) -> int:
            if needle == '':
                return 0
            next_arr = self.getNext(needle)
            print(next_arr)
            j = -1
            for i in range(0, len(haystack)):
                while( j >= 0 and haystack[i] != needle[j+1]):
                    j = next_arr[j]
                if haystack[i] == needle[j+1]:
                    j += 1
                if j == (len(needle) - 1):
                    return (i - len(needle) + 1)
            return - 1
    ```

    

### 重复的子字符串

看看自己的提交记录吧

* kmp的思路，利用kmp的最大前后缀数组，用最后一位的值可以被整除来实现
* 利用(s+s).find(s, 1) != len(s)， 如果是循环的，那么找到s则必然不需要在第二个s的地方找到



### python 中的栈和队列

* 栈

  * 直接使用list即可

  * 常见的函数:

  * ```python
    stack = list()
    stack.append(x)
    stack.pop(x)
    stack.empty()
    
    ```

* 队列

  * import queue

  * ```python
    import queue
    q = queue.Queue()
    q.put(x)
    q.get(x)
    q.empty()
    ```

    

### 逆波兰表达式

* 使用栈就行
* 可以使用map来替代switch

```javascript
/**
 * @param {string[]} tokens
 * @return {number}
 */
var evalRPN = function(tokens) {
    const s = new Map([
        ['+', (a, b) => a*1 + b*1],
        ['-', (a, b) => b - a],
        ['*', (a, b) => b * a],
        ["/", (a, b) => (b/a) | 0]
    ]);
    const stack = []
    for (const i of tokens) {
        if (!s.has(i)) {
            stack.push(i)
            continue;
        }
        stack.push(s.get(i)(stack.pop(), stack.pop()))
    }
    return stack.pop()
};
```



### 滑动窗口最大值

* 使用队列
* 需要满足每次获得队头元素都是最大值
* push修改，如果队列入口的值小于要push的值，那么要弹出这个值，直到遇到大于或者空为止。

```javascript
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
var maxSlidingWindow = function(nums, k) {
    // 保存的是下标
    const q = [];
    const ans = [];
    for (let i = 0; i<nums.length; i++) {
        // 如果队列入口的值小于要push的值，那么就弹出，直到大于为止
        while (q.length && nums[i] >= nums[q[q.length - 1]]) {
            q.pop();
        }
        q.push(i);
        // 判断当前的最大值是否在窗口中，不在则出队
        if (q[0] < i-k+1) {
            q.shift();
        }
        // 起码到达了窗口长度开始添加答案
        if (i>=k-1) ans.push(nums[q[0]]);
    }
    return ans
};
```



### [ 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

* 统计：使用map

* 排序：优先级队列

* ```javascript
  function QueueElement(element, priority) {
      this.element = element;
      this.priority = priority;
  }
  
  function PriorityQueue() {
      this.items = [];
  }
  
  PriorityQueue.prototype.enQueue = function(element, priority) {
      let queueElement = new QueueElement(element, priority);
      if (this.items.length == 0) {
          this.items.push(queueElement)
      } else {
          let flag = false;
          for(let i = 0; i<this.items.length; i++) {
              if(queueElement.priority > this.items[i].priority) {
                  this.items.splice(i, 0, queueElement);
                  flag = true;
                  break;
              }
          }
          if (!flag) {
              this.items.push(queueElement);
          }
      }
  }
  
  PriorityQueue.prototype.pop = function() {
      return this.items.shift();
  }
  
  PriorityQueue.prototype.front = function() {
      return this.items[0];
  }
  
  ```

  

* 输出

```javascript
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */

function QueueElement(element, priority) {
    this.element = element;
    this.priority = priority;
}

function PriorityQueue() {
    this.items = [];
}

PriorityQueue.prototype.enQueue = function(element, priority) {
    let queueElement = new QueueElement(element, priority);
    if (this.items.length == 0) {
        this.items.push(queueElement)
    } else {
        let flag = false;
        for(let i = 0; i<this.items.length; i++) {
            if(queueElement.priority > this.items[i].priority) {
                this.items.splice(i, 0, queueElement);
                flag = true;
                break;
            }
        }
        if (!flag) {
            this.items.push(queueElement);
        }
    }
}

PriorityQueue.prototype.pop = function() {
    return this.items.shift();
}

PriorityQueue.prototype.front = function() {
    return this.items[0];
}

var topKFrequent = function(nums, k) {
    let count_map = new Map();
    let pri_queue = new PriorityQueue();
    let ans = [];
    for(let i in nums) {
        if(count_map.has(nums[i])) {
            let tmp = count_map.get(nums[i]);
            count_map.set(nums[i], tmp + 1)
        } else {
            count_map.set(nums[i], 1)
        }
    }

    for (let [key, value] of count_map) {
        pri_queue.enQueue(key, value);
    }

    for(let i = 0; i<k; i++) {
        ans.push(pri_queue.pop().element)
    }

    return ans;
};
```



### 二叉树的遍历

* 迭代遍历

* 分为前序后序和中序两种

* ```javascript
  // 前序
  /**
   * Definition for a binary tree node.
   * function TreeNode(val, left, right) {
   *     this.val = (val===undefined ? 0 : val)
   *     this.left = (left===undefined ? null : left)
   *     this.right = (right===undefined ? null : right)
   * }
   */
  /**
   * @param {TreeNode} root
   * @return {number[]}
   */
  
  // 使用栈，然后先进right子树，因为栈是先进后出，然后进left子树
  
  var preorderTraversal = function(root, res = []) {
      if(!root) {
          return res;
      }
      const stack = [root];
      let cur = null;
      while(stack.length) {
          cur = stack.pop();
          res.push(cur.val);
          if (cur.right) {
              stack.push(cur.right);
          }
          if(cur.left) {
              stack.push(cur.left);
          }
      }
      return res;
  };
  
  // 后序
  // 和前序一样，区别就是添加了最后的翻转，因为 中右左 翻转就是 左右中 
  var postorderTraversal = function(root, ans = []) {
      if (!root) return ans;
      const stack = [root];
      let cur = null;
      while(stack.length) {
          cur = stack.pop();
          ans.push(cur.val);
          if (cur.left) {
              stack.push(cur.left);
          }
          if(cur.right) {
              stack.push(cur.right);
          }
      }
      return ans.reverse();
  };
  
  // 中序
  // 不一样，使用一个指针一直盯着，到不能继续下去才开始输出
  var mid = function(root, ans = []) {
      const stack = [];
      let cur = root;
      while(cur != null || stack.length) {
          if (cur != null) {
              stack.push(cur);
              cur = cur.left;
          } else {
              cur = stack.pop();
              ans.push(cur.val);
              cur = cur.right;
          }
      }
  }
  
  var inorderTraversal = function(root) {
      let ans = [];
      mid(root, ans);
      return ans;
  };
  ```



* 递归，非常容易，直接写就行了

* ```javascript
  
  // 中序
  var mid = function(root, ans = []) {
      if (!root) {
          return;
      }
      mid(root.left, ans);
      ans.push(root.val);
      mid(root.right, ans);
  }
  
  var inorderTraversal = function(root) {
      let ans = [];
      mid(root, ans);
      return ans;
  };
  
  // 后序
  var back = function(root = new TreeNode(), ans = []) {
      if (!root) {
          return
      }
      back(root.left, ans);
      back(root.right, ans);
      ans.push(root.val);
  
  }
  
  var postorderTraversal = function(root) {
      let ans = []
      back(root, ans);
      return ans;
  };
  
  // 前序
  var front = function(root, ans = []) {
      if (!root ) {
          return
      }
      ans.push(root.val);
      front(root.left, ans);
      front(root.right, ans);
  
  }
  
  var preorderTraversal = function(root) {
      let ans = [];
      front(root, ans);
      return ans;
  };
  ```



* **层级遍历**

* ```javascript
  /**
   * Definition for a binary tree node.
   * function TreeNode(val, left, right) {
   *     this.val = (val===undefined ? 0 : val)
   *     this.left = (left===undefined ? null : left)
   *     this.right = (right===undefined ? null : right)
   * }
   */
  /**
   * @param {TreeNode} root
   * @return {number[][]}
   */
  var levelOrder = function(root) {
      let res = [], queue = [root];
      if (!root) return res;
      while(queue.length) {
          // 记录当前层级节点数
          let len_of_the_level = queue.length;
          //存放每一层的节点 
          let level_nodes = [];
          for(let i=0;i<len_of_the_level;i++) {
              let node = queue.shift();
              level_nodes.push(node.val);
              // 存放当前层下一层的节点
              node.left && queue.push(node.left);
              node.right && queue.push(node.right);
          }
          //把每一层的结果放到结果数组
          res.push(level_nodes);
      }
      return res;
  };
  ```

  



### 回溯法

```
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}

```



### 组合

典型的回溯问题

* 每一层迭代等于一个循环，因为无法手动写出那么多的循环
* 对于每一次迭代使用for循环遍历这一层
* 终止条件就是path数组里面的长度和目标一致。
* 使用展开运算符，**否则最终ret中的每一个数组都是指向一个地址**，所以会全是空

可以剪枝

* 通过n - i + 1 > k - len(path)可以剪枝 

```js
/**
 * @param {number} n
 * @param {number} k
 * @return {number[][]}
 */
let ret = [];
let path = [];

let back = function(n, k, startIndex) {
    if (path.length == k) {
        ret.push([...path]);
        return;
    }
    let tmp_len = path.length
    for (let i=startIndex; i<=n-k+tmp_len+1; i++) {
        path.push(i);
        back(n, k, i+1);
        path.pop();
    }
}

var combine = function(n, k) {
    ret = [];
    back(n, k, 1);
    return ret;
};

```

