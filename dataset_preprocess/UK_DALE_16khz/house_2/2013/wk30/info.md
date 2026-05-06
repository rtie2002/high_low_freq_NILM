# UK-DALE House 2: 2013 Week 30 Data Info

## 1. File Naming Convention
The high-frequency files are named using the following format:
`vi-[UNIX_TIMESTAMP]_[MICROSECONDS].flac`

*   **UNIX_TIMESTAMP**: Seconds elapsed since January 1, 1970 (UTC).
*   **MICROSECONDS**: Sub-second precision.

## 2. Timezone Context (July 2013)
*   **UTC**: Coordinated Universal Time (Base).
*   **UK Time (BST)**: British Summer Time (**UTC + 1 hour**).

## 3. Data Range for Week 30

### Start of Dataset
*   **File**: `vi-1374447600_095189.flac`
*   **Unix Timestamp**: `1374447600`
*   **UTC Time**: 2013-07-21 23:00:00
*   **UK Local Time (BST)**: **2013-07-22 00:00:00 (Monday Midnight)**

### End of Dataset (Last File Start)
*   **File**: `vi-1375048800_819257.flac`
*   **Unix Timestamp**: `1375048800`
*   **UTC Time**: 2013-07-28 22:00:00
*   **UK Local Time (BST)**: **2013-07-28 23:00:00 (Sunday Night)**

## 4. How to Convert Manually
To convert any timestamp in this dataset:
1.  **Find the Timestamp**: Take the 10 digits after `vi-`.
2.  **UTC Conversion**: Use Python `datetime.fromtimestamp(ts, tz=datetime.timezone.utc)`.
3.  **Local Alignment**: Add **1 hour** to the UTC result to get the actual time recorded in the UK home during summer.

---
*Note: This week provides a perfect 7-day alignment starting from the first second of Monday.*
