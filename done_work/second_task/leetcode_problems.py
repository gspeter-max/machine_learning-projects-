''' 
2.1093. Statistics from a Large Sample, 
3. 2241. Design an ATM Machine, 

'''

# 2. Solution 
count = [1,1,1,1,2,2,2,3,3,4,4]

class Solution:
    def sampleStats(self, count): 
        
        _min = 10000000000000000000000000000000000000
        _max = 0
        sums = 0 
        for valeus in count:
            sums  = sums + valeus 
            if valeus > 0: 
                if valeus < _min: 
                    _min = valeus
                if valeus > _max : 
                    _max = valeus 
        
        def median(count):
            
            if len(count) %  2 == 0: 
                
                temp_index = int((len(count) + 1) / 2) 
                median_index = [temp_index , temp_index + 1]
                temp_ = 0 
                for v in median_index: 
                    temp_ = temp_ + count[v]
                return temp_ / 2 
            else: 
                index = int((len(count) +  1) / 2)
                return count[index]
            
            
        def mode(x): 
            y = {} 
            for a in x: 
                if a not in y: 
                    y[a] = 1 
                else: 
                    y[a] += 1 
                
            result = [g for g, l in y.items() if l == max(y.values())] 
            return result[0]
        
        
        return _min, _max , sums/ len(count), median(count) , mode(count)
    
                     
''' atm machine '''

class ATM:

    def __init__(self):
        self.denominates = [0,0,0,0,0]
        self.banknotes = [20,50,100,200,500]

    def deposit(self, banknotesCount: List[int]) -> None:
        
        for i in range(len(self.denominates)): 
            self.denominates[i] +=  banknotesCount[i]

    def withdraw(self, amount: int) -> List[int]:
        
        left_amount = amount 
        used_notes = [0]*5 
        
        for i in range(4,-1,-1):  
            if left_amount >= self.banknotes[i]:
            
                need = left_amount // self.banknotes[i]
                used_notes[i] = min(self.denominates[i],need)
                left_amount  = left_amount - (self.banknotes[i] * used_notes[i])
            
        if left_amount > 0: 
            return [-1]            
        
        for i in range(len(used_notes)) : 
            self.denominates[i] -= used_notes[i]
        
        return used_notes 
    

# Your ATM object will be instantiated and called as such:
# obj = ATM()
# obj.deposit(banknotesCount)
# param_2 = obj.withdraw(amount)