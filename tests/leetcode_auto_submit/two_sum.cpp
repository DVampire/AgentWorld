/*
 * @lc app=leetcode id=1 lang=cpp
 *
 * [1] Two Sum
 */

// @lc code=start
class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            map<int,int> mp;
            vector<int> ans;
            for(int i=0;i<nums.size();i++)
            {
                if(mp.find(target-nums[i])!=mp.end())
                {
                    ans.push_back(i);
                    ans.push_back(mp[target-nums[i]]);
                    return ans;
                }
                else
                    mp[nums[i]]=i;
            }
            return ans; 
        }
    };
    // @lc code=end
    
    