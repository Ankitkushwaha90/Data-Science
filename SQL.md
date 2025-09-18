
# Complete SQL Notes

**Author:** Grok AI  
**Date:** September 18, 2025, 11:20 AM IST  

This document provides a comprehensive guide to SQL (Structured Query Language), covering its fundamentals, key commands, and practical examples. SQL is essential for managing and querying relational databases, widely used in data science, backend development, and analytics. These notes are structured for beginners to advanced learners, with examples using a sample database.

---

## Introduction to SQL

### What is SQL?
- **Definition**: SQL is a standard language for storing, manipulating, and retrieving data in relational database management systems (RDBMS) like MySQL, PostgreSQL, SQLite, and Oracle.
- **Purpose**: Enables data definition (DDL), data manipulation (DML), and data control (DCL).
- **Key Features**:
  - Declarative: Specify what you want, not how to get it.
  - Case-insensitive (e.g., `SELECT` = `select`).
  - Supports ACID properties (Atomicity, Consistency, Isolation, Durability).

### Sample Database
We’ll use a simple database with two tables:
- **Employees**:
  | ID  | Name       | Department | Salary  | Hire_Date  |
  |-----|------------|------------|---------|------------|
  | 1   | Alice      | HR         | 50000   | 2023-01-15 |
  | 2   | Bob        | IT         | 60000   | 2022-06-10 |
  | 3   | Charlie    | IT         | 55000   | 2023-03-20 |
  | 4   | Diana      | Finance    | 65000   | 2022-11-05 |

- **Departments**:
  | Dept_ID | Dept_Name | Location    |
  |---------|-----------|-------------|
  | 1       | HR        | New York    |
  | 2       | IT        | Bangalore   |
  | 3       | Finance   | London      |

---

## SQL Basics

### 1. Data Definition Language (DDL)
Used to define and modify database structures.

- **CREATE TABLE**:
  Creates a new table.
  ```sql
  CREATE TABLE Employees (
      ID INT PRIMARY KEY,
      Name VARCHAR(50),
      Department VARCHAR(50),
      Salary DECIMAL(10, 2),
      Hire_Date DATE
  );
  ```

- **ALTER TABLE**:
  Modifies an existing table.
  ```sql
  ALTER TABLE Employees
  ADD COLUMN Email VARCHAR(100);
  ```

- **DROP TABLE**:
  Deletes a table.
  ```sql
  DROP TABLE Employees;
  ```

- **TRUNCATE TABLE**:
  Removes all data but keeps the table structure.
  ```sql
  TRUNCATE TABLE Employees;
  ```

---

### 2. Data Manipulation Language (DML)
Used to manipulate data within tables.

- **INSERT**:
  Adds new rows.
  ```sql
  INSERT INTO Employees (ID, Name, Department, Salary, Hire_Date)
  VALUES (5, 'Eve', 'HR', 52000, '2024-01-10');
  ```

- **SELECT**:
  Retrieves data.
  ```sql
  SELECT Name, Salary
  FROM Employees;
  ```
  **Output**:
  | Name    | Salary |
  |---------|--------|
  | Alice   | 50000  |
  | Bob     | 60000  |
  | Charlie | 55000  |
  | Diana   | 65000  |
  | Eve     | 52000  |

- **UPDATE**:
  Modifies existing data.
  ```sql
  UPDATE Employees
  SET Salary = 62000
  WHERE ID = 2;
  ```

- **DELETE**:
  Removes rows.
  ```sql
  DELETE FROM Employees
  WHERE ID = 5;
  ```

---

### 3. Data Query Language (DQL)
Focuses on querying data.

- **SELECT with WHERE**:
  Filters rows.
  ```sql
  SELECT Name, Department
  FROM Employees
  WHERE Salary > 55000;
  ```
  **Output**:
  | Name   | Department |
  |--------|------------|
  | Bob    | IT         |
  | Diana  | Finance    |

- **SELECT with ORDER BY**:
  Sorts results.
  ```sql
  SELECT Name, Salary
  FROM Employees
  ORDER BY Salary DESC;
  ```
  **Output**:
  | Name   | Salary |
  |--------|--------|
  | Diana  | 65000  |
  | Bob    | 60000  |
  | Charlie| 55000  |
  | Alice  | 50000  |

- **SELECT with DISTINCT**:
  Removes duplicates.
  ```sql
  SELECT DISTINCT Department
  FROM Employees;
  ```
  **Output**:
  | Department |
  |------------|
  | HR         |
  | IT         |
  | Finance    |

---

## Advanced SQL Queries

### 1. Joins
Combine data from multiple tables.

- **INNER JOIN**:
  Matches rows with common values.
  ```sql
  SELECT e.Name, d.Location
  FROM Employees e
  INNER JOIN Departments d ON e.Department = d.Dept_Name;
  ```
  **Output**:
  | Name    | Location  |
  |---------|-----------|
  | Alice   | New York  |
  | Bob     | Bangalore |
  | Charlie | Bangalore |
  | Diana   | London    |

- **LEFT JOIN**:
  Includes all rows from the left table.
  ```sql
  SELECT e.Name, d.Location
  FROM Employees e
  LEFT JOIN Departments d ON e.Department = d.Dept_Name;
  ```

- **RIGHT JOIN**:
  Includes all rows from the right table.
  ```sql
  SELECT e.Name, d.Location
  FROM Employees e
  RIGHT JOIN Departments d ON e.Department = d.Dept_Name;
  ```

- **FULL JOIN**:
  Includes all rows from both tables.
  ```sql
  SELECT e.Name, d.Location
  FROM Employees e
  FULL JOIN Departments d ON e.Department = d.Dept_Name;
  ```

---

### 2. Aggregate Functions
Perform calculations on data.

- **COUNT**:
  Counts rows.
  ```sql
  SELECT Department, COUNT(*) as Employee_Count
  FROM Employees
  GROUP BY Department;
  ```
  **Output**:
  | Department | Employee_Count |
  |------------|----------------|
  | HR         | 2              |
  | IT         | 2              |
  | Finance    | 1              |

- **SUM**, **AVG**, **MAX**, **MIN**:
  ```sql
  SELECT SUM(Salary) as Total_Salary, AVG(Salary) as Avg_Salary
  FROM Employees;
  ```
  **Output**:
  | Total_Salary | Avg_Salary |
  |--------------|------------|
  | 230000       | 57500      |

- **GROUP BY with HAVING**:
  Filters grouped results.
  ```sql
  SELECT Department, AVG(Salary) as Avg_Salary
  FROM Employees
  GROUP BY Department
  HAVING AVG(Salary) > 55000;
  ```
  **Output**:
  | Department | Avg_Salary |
  |------------|------------|
  | IT         | 57500      |
  | Finance    | 65000      |

---

### 3. Subqueries and Nested Queries

- **Subquery in SELECT**:
  ```sql
  SELECT Name, (SELECT Location FROM Departments WHERE Dept_Name = e.Department) as Location
  FROM Employees e;
  ```

- **Subquery in WHERE**:
  ```sql
  SELECT Name, Salary
  FROM Employees
  WHERE Salary > (SELECT AVG(Salary) FROM Employees);
  ```
  **Output**:
  | Name   | Salary |
  |--------|--------|
  | Bob    | 60000  |
  | Diana  | 65000  |

- **EXISTS**:
  ```sql
  SELECT Name
  FROM Employees e
  WHERE EXISTS (SELECT 1 FROM Departments d WHERE d.Dept_Name = e.Department);
  ```

---

### 4. Indexes and Performance

- **CREATE INDEX**:
  Improves query speed.
  ```sql
  CREATE INDEX idx_dept ON Employees(Department);
  ```

- **DROP INDEX**:
  Removes an index.
  ```sql
  DROP INDEX idx_dept;
  ```

---

## Data Control Language (DCL)

- **GRANT**:
  Assigns permissions.
  ```sql
  GRANT SELECT ON Employees TO 'user'@'localhost';
  ```

- **REVOKE**:
  Removes permissions.
  ```sql
  REVOKE SELECT ON Employees FROM 'user'@'localhost';
  ```

---

## Transaction Control Language (TCL)

- **COMMIT**:
  Saves changes.
  ```sql
  COMMIT;
  ```

- **ROLLBACK**:
  Undoes changes.
  ```sql
  ROLLBACK;
  ```

- **SAVEPOINT**:
  Sets a savepoint.
  ```sql
  SAVEPOINT savepoint1;
  ```

---

## Practical Examples

### 1. Employee Salary Report
```sql
SELECT Department, AVG(Salary) as Avg_Salary, COUNT(*) as Employee_Count
FROM Employees
GROUP BY Department
HAVING COUNT(*) > 1
ORDER BY Avg_Salary DESC;
```
**Output**:
| Department | Avg_Salary | Employee_Count |
|------------|------------|----------------|
| IT         | 57500      | 2              |
| HR         | 51000      | 2              |

### 2. Join with Aggregation
```sql
SELECT d.Location, COUNT(e.ID) as Employee_Count
FROM Departments d
LEFT JOIN Employees e ON d.Dept_Name = e.Department
GROUP BY d.Location;
```
**Output**:
| Location  | Employee_Count |
|-----------|----------------|
| New York  | 2              |
| Bangalore | 2              |
| London    | 1              |

---

## Best Practices

- **Naming Conventions**: Use clear, consistent names (e.g., `tbl_employees`).
- **Indexing**: Index frequently queried columns.
- **Backup**: Regularly back up databases.
- **Security**: Use parameterized queries to prevent SQL injection.

---

## Conclusion

SQL is a powerful tool for data management. Mastering DDL, DML, DQL, DCL, and TCL enables efficient database operations. Practice with real datasets (e.g., Kaggle) to solidify skills.

### References
- MySQL Documentation: https://dev.mysql.com/doc/
- PostgreSQL Tutorial: https://www.postgresqltutorial.com/
- W3Schools SQL: https://www.w3schools.com/sql/
---
```
