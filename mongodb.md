# Complete MongoDB Notes

**Author:** Grok AI  
**Date:** September 18, 2025, 11:23 AM IST  

This document provides a comprehensive guide to MongoDB, a popular NoSQL database that uses a document-oriented model. Unlike relational databases, MongoDB stores data in JSON-like documents (BSON format), making it flexible for handling unstructured or semi-structured data. These notes are designed for beginners to advanced users, with examples using a sample dataset.

---

## Introduction to MongoDB

### What is MongoDB?
- **Definition**: MongoDB is a distributed, open-source NoSQL database designed for scalability and flexibility.
- **Key Features**:
  - Document-based: Stores data in BSON (Binary JSON) documents.
  - Schema-less: No fixed schema, allowing dynamic fields.
  - Scalability: Supports horizontal scaling via sharding.
  - High Performance: Uses in-memory processing and indexing.
- **Use Cases**: Content management, real-time analytics, e-commerce, IoT.

### Sample Dataset
We’ll use a sample database named `companyDB` with two collections:
- **employees**:
  ```json
  [
    { "_id": 1, "name": "Alice", "department": "HR", "salary": 50000, "hireDate": "2023-01-15", "skills": ["HR", "Recruitment"] },
    { "_id": 2, "name": "Bob", "department": "IT", "salary": 60000, "hireDate": "2022-06-10", "skills": ["Python", "DevOps"] },
    { "_id": 3, "name": "Charlie", "department": "IT", "salary": 55000, "hireDate": "2023-03-20", "skills": ["Java", "Cloud"] },
    { "_id": 4, "name": "Diana", "department": "Finance", "salary": 65000, "hireDate": "2022-11-05", "skills": ["Finance", "Audit"] }
  ]
  ```
- **departments**:
  ```json
  [
    { "_id": 1, "deptName": "HR", "location": "New York" },
    { "_id": 2, "deptName": "IT", "location": "Bangalore" },
    { "_id": 3, "deptName": "Finance", "location": "London" }
  ]
  ```

---

## MongoDB Basics

### 1. Installation and Setup
- **Install MongoDB**: Download from [mongodb.com](https://www.mongodb.com/try/download/community) or use Docker.
  ```bash
  docker run -d -p 27017:27017 --name mongodb mongo:latest
  ```
- **Connect**: Use MongoDB Shell (`mongosh`) or a driver (e.g., PyMongo in Python).
  ```javascript
  mongosh "mongodb://localhost:27017/companyDB"
  ```

### 2. Core Concepts
- **Database**: Container for collections (e.g., `companyDB`).
- **Collection**: Group of documents (e.g., `employees`).
- **Document**: Single record, similar to a row in SQL (e.g., `{ _id: 1, name: "Alice" }`).
- **Field**: Key-value pair within a document (e.g., `name: "Alice"`).

---

## CRUD Operations

### 1. Create Operations

- **Insert One Document**:
  ```javascript
  db.employees.insertOne({
      _id: 5,
      name: "Eve",
      department: "HR",
      salary: 52000,
      hireDate: "2024-01-10",
      skills: ["HR", "Training"]
  });
  ```

- **Insert Many Documents**:
  ```javascript
  db.employees.insertMany([
      { _id: 6, name: "Frank", department: "IT", salary: 58000, hireDate: "2023-05-15", skills: ["SQL", "AWS"] },
      { _id: 7, name: "Grace", department: "Finance", salary: 63000, hireDate: "2022-09-01", skills: ["Excel", "Budgeting"] }
  ]);
  ```

---

### 2. Read Operations

- **Find All Documents**:
  ```javascript
  db.employees.find();
  ```
  **Output** (sample):
  ```json
  { "_id": 1, "name": "Alice", "department": "HR", "salary": 50000, "hireDate": "2023-01-15", "skills": ["HR", "Recruitment"] }
  { "_id": 2, "name": "Bob", "department": "IT", "salary": 60000, "hireDate": "2022-06-10", "skills": ["Python", "DevOps"] }
  ```

- **Find with Query**:
  ```javascript
  db.employees.find({ salary: { $gt: 55000 } });
  ```
  **Output**:
  ```json
  { "_id": 2, "name": "Bob", "department": "IT", "salary": 60000, "hireDate": "2022-06-10", "skills": ["Python", "DevOps"] }
  { "_id": 4, "name": "Diana", "department": "Finance", "salary": 65000, "hireDate": "2022-11-05", "skills": ["Finance", "Audit"] }
  { "_id": 6, "name": "Frank", "department": "IT", "salary": 58000, "hireDate": "2023-05-15", "skills": ["SQL", "AWS"] }
  { "_id": 7, "name": "Grace", "department": "Finance", "salary": 63000, "hireDate": "2022-09-01", "skills": ["Excel", "Budgeting"] }
  ```

- **Projection (Limit Fields)**:
  ```javascript
  db.employees.find({ department: "IT" }, { name: 1, salary: 1, _id: 0 });
  ```
  **Output**:
  ```json
  { "name": "Bob", "salary": 60000 }
  { "name": "Charlie", "salary": 55000 }
  { "name": "Frank", "salary": 58000 }
  ```

- **Sort**:
  ```javascript
  db.employees.find().sort({ salary: -1 }); // -1 for descending, 1 for ascending
  ```

- **Limit**:
  ```javascript
  db.employees.find().limit(2);
  ```

---

### 3. Update Operations

- **Update One Document**:
  ```javascript
  db.employees.updateOne(
      { _id: 2 },
      { $set: { salary: 62000, skills: ["Python", "DevOps", "Docker"] } }
  );
  ```

- **Update Many Documents**:
  ```javascript
  db.employees.updateMany(
      { department: "IT" },
      { $inc: { salary: 2000 } } // Increases salary by 2000
  );
  ```

- **Upsert (Insert if Not Exists)**:
  ```javascript
  db.employees.updateOne(
      { _id: 8 },
      { $set: { name: "Hank", department: "IT", salary: 59000, hireDate: "2023-07-01", skills: ["Go", "Kubernetes"] } },
      { upsert: true }
  );
  ```

---

### 4. Delete Operations

- **Delete One Document**:
  ```javascript
  db.employees.deleteOne({ _id: 5 });
  ```

- **Delete Many Documents**:
  ```javascript
  db.employees.deleteMany({ salary: { $lt: 55000 } });
  ```

- **Drop Collection**:
  ```javascript
  db.employees.drop();
  ```

---

## Advanced Queries

### 1. Joins (Lookup)
MongoDB uses `$lookup` for joining collections.

```javascript
db.employees.aggregate([
  {
    $lookup: {
      from: "departments",
      localField: "department",
      foreignField: "deptName",
      as: "dept_details"
    }
  }
]);
```
**Output** (sample):
```json
{
  "_id": 1,
  "name": "Alice",
  "department": "HR",
  "salary": 50000,
  "hireDate": "2023-01-15",
  "skills": ["HR", "Recruitment"],
  "dept_details": [{ "_id": 1, "deptName": "HR", "location": "New York" }]
}
```

### 2. Aggregation Pipeline

- **Group**:
  ```javascript
  db.employees.aggregate([
    { $group: { _id: "$department", avgSalary: { $avg: "$salary" } } }
  ]);
  ```
  **Output**:
  ```json
  { "_id": "HR", "avgSalary": 51000 }
  { "_id": "IT", "avgSalary": 58333.33 }
  { "_id": "Finance", "avgSalary": 64000 }
  ```

- **Match and Project**:
  ```javascript
  db.employees.aggregate([
    { $match: { salary: { $gt: 55000 } } },
    { $project: { name: 1, salary: 1, _id: 0 } }
  ]);
  ```

### 3. Indexes
- **Create Index**:
  ```javascript
  db.employees.createIndex({ department: 1 });
  ```

- **Drop Index**:
  ```javascript
  db.employees.dropIndex({ department: 1 });
  ```

---

## Administration Commands

- **Create Database**:
  ```javascript
  use companyDB;
  ```

- **Show Databases**:
  ```javascript
  show databases;
  ```

- **Drop Database**:
  ```javascript
  db.dropDatabase();
  ```

- **Show Collections**:
  ```javascript
  show collections;
  ```

---

## Practical Examples

### 1. Employee Salary Report
```javascript
db.employees.aggregate([
  { $group: { _id: "$department", totalSalary: { $sum: "$salary" }, count: { $sum: 1 } } },
  { $sort: { totalSalary: -1 } }
]);
```
**Output**:
```json
{ "_id": "Finance", "totalSalary": 128000, "count": 2 }
{ "_id": "IT", "totalSalary": 175000, "count": 3 }
{ "_id": "HR", "totalSalary": 102000, "count": 2 }
```

### 2. Find Employees with Specific Skills
```javascript
db.employees.find({ skills: "Python" });
```
**Output**:
```json
{ "_id": 2, "name": "Bob", "department": "IT", "salary": 60000, "hireDate": "2022-06-10", "skills": ["Python", "DevOps", "Docker"] }
```

---

## Best Practices

- **Schema Design**: Use embedded documents for related data to reduce joins.
- **Indexing**: Index frequently queried fields (e.g., `department`).
- **Security**: Enable authentication and use role-based access control (RBAC).
- **Backup**: Use `mongodump` for backups.
  ```bash
  mongodump --db companyDB --out /backup
  ```

---

## Conclusion

MongoDB is a powerful NoSQL solution for handling flexible, scalable data. Mastering CRUD, aggregation, and administration commands enables efficient data management. Practice with real-world datasets (e.g., MongoDB University) to solidify skills.

### References
- MongoDB Documentation: https://docs.mongodb.com/
- MongoDB University: https://university.mongodb.com/
- PyMongo Tutorial: https://pymongo.readthedocs.io/
---
```
