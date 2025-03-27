'''
📌 THURSDAY (March 28) – SQL for Machine Learning
🎯 Tasks:

Implement recursive CTEs for hierarchical data.

Design a SQL-based feature store (storing & retrieving ML features).

Optimize complex queries using EXPLAIN ANALYZE.

🛑 Hard Mode: Create a real-time ML pipeline using SQL & Apache Kafka.
'''


CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(100),
    manager_id INT NULL
);

INSERT INTO employees VALUES 
(1, 'Alice', NULL),      -- CEO (No manager)
(2, 'Bob', 1),           -- Reports to Alice
(3, 'Charlie', 1),       -- Reports to Alice
(4, 'David', 2),         -- Reports to Bob
(5, 'Eve', 2),           -- Reports to Bob
(6, 'Frank', 3);         -- Reports to Charlie


create index idx_manager on employees(manager_id)
explain analyze 
with recursive temp_temp as ( 
    select 
        emp_id,
        emp_name, 
        manager_id , 
        1 as levels 
    from employees
    where manager_id is NULL

    union all 

    select a1.emp_id, 
        a1.emp_name, 
        a1.manager_id, 
        levels + 1 as levels 

    from  employees a1 
    join temp_temp a2 on a2.emp_id = a1.manager_id 

) 
select * from temp_temp ; 





CREATE TABLE user_features (
    user_id INT PRIMARY KEY,
    avg_session_time FLOAT,
    total_purchases INT,
    last_login_date DATE
);

INSERT INTO user_features VALUES 
(101, 15.2, 10, '2024-03-25'),
(102, 8.5, 2, '2024-03-24'),
(103, 12.3, 5, '2024-03-27'),
(104, 20.1, 15, '2024-03-26');


explain analyze 
with temp_temp as (
     
--Identify users whose total purchases are greater than the average total purchases.
--Calculate the difference between the current time and the last login date.
--Rank users based on their total spending.
--Create a binary mask indicating whether a user has exceeded the average session time.
--Order the table based on the ranking system.
    
    select 
        case when total_purchase > ( select avg(total_purchase) from user_features) then 1  else 0 end as avg_purchase_acsseder ,
        datediff(CURRENT_ROW , last_login_date) as spending_time, 
        row_number() over( order by total_purchases desc) as ranking,
        case when avg_session_time > (select avg(avg_session_time) from user_features ) then 1 else 0 end as avg_session_acsseder
    
    from user_features

) 
select * from temp_temp
order by ranking ; 



CREATE TABLE transactions (
    txn_id INT PRIMARY KEY,
    user_id INT,
    amount FLOAT,
    txn_date TIMESTAMP,
    category VARCHAR(50)
);

INSERT INTO transactions VALUES 
(1001, 101, 120.5, '2024-03-25 14:30:00', 'Electronics'),
(1002, 102, 45.0, '2024-03-24 09:15:00', 'Groceries'),
(1003, 101, 250.7, '2024-03-27 18:45:00', 'Fashion'),
(1004, 103, 85.2, '2024-03-26 12:10:00', 'Books');



create index idx_index on transactions(category);
create index idx_user  on  transactions(user_id);
with temp_temp as (

-- First, group by category and calculate the sum and average of the amount.
-- Apply a 7-day rolling sum window.
-- Compute the user's average within each partition.
-- Determine the difference between the first and the last transaction for each user.

    select *,
        
        avg(amount) over( partition by category ) as avg_amount_category, 
        sum(amount) over( partition by category ) as sum_amount_category, 
        sum(amount) over(partition by user_id order by txn_date rows between  6 preceding and current row ) as 7_day_rolling, 
        avg(amount) over(partition by user_id ) as user_average, 
        first_value(txn_date) over( partition by user_id order by txn_date) as first_date , 
        last_value(txn_date) over(partition by  user_id order by txn_date) as last_date 

    from transactions 

) 
select avg_amount_category,
    sum_amount_category, 
    7_day_rolling,
    user_average,
    datediff(last_date, first_date) as date_diff 
from temp_temp ; 


create table if not exists updated_table as (
    select *, avg_amount_category,
        sum_amount_category, 
        7_day_rolling,
        user_average,
        datediff(last_date, first_date) as date_diff 
    from temp_temp  
); 

with outliers_detection as ( 
    select *, 
        stddev(amount) over(partition by user_id ) as stddev_users
    from updated_table
) 
select * 
from  outliers_detection 
where amount > user_average + 3 * stddev_users or amount < user_average - 3 * stddev_users ; 
