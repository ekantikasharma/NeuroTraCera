import asyncio
import json
import os
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# SQLITE DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = Path(
    os.getenv(
        "SQLITE_DB_PATH",
        str(BASE_DIR / "brain_tumor_xai.db")
    )
)


# ============================================================
# DATABASE CLASS
# ============================================================

class Database:

    def __init__(self, db_path: Optional[str] = None):

        self.db_path = Path(
            db_path or DB_PATH
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # CONNECTION
    # ========================================================

    def _get_connection(self):

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        return connection


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def _initialize_database(self):

        connection = self._get_connection()

        try:

            # =================================================
            # USERS TABLE
            # =================================================

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    login_id TEXT NOT NULL UNIQUE,

                    password_hash TEXT NOT NULL,

                    password_salt TEXT,

                    patient_name TEXT,

                    patient_id TEXT,

                    created_at TEXT NOT NULL,

                    last_login TEXT
                )
                """
            )


            # =================================================
            # ANALYSES TABLE
            # =================================================

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    user_id TEXT,

                    patient_case_id TEXT,

                    patient_age INTEGER,

                    patient_gender TEXT,

                    created_at TEXT NOT NULL,

                    document TEXT NOT NULL
                )
                """
            )


            # =================================================
            # REPORTS TABLE
            # =================================================

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (

                    id TEXT PRIMARY KEY,

                    analysis_id TEXT NOT NULL,

                    user_id TEXT,

                    patient_case_id TEXT,

                    patient_id TEXT,

                    findings TEXT NOT NULL,

                    impression TEXT NOT NULL,

                    report_version INTEGER DEFAULT 1,

                    report_content TEXT NOT NULL,

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL
                )
                """
            )


            # =================================================
            # INDEXES
            # =================================================

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_users_login_id
                ON users(login_id)
                """
            )


            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_analyses_user_id
                ON analyses(user_id)
                """
            )


            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_analyses_created_at
                ON analyses(created_at)
                """
            )


            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_analyses_patient_case_id
                ON analyses(patient_case_id)
                """
            )


            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_reports_analysis_id
                ON reports(analysis_id)
                """
            )


            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_reports_user_id
                ON reports(user_id)
                """
            )


            connection.commit()

        finally:

            connection.close()


    # ========================================================
    # CONNECT / INITIALIZE
    # ========================================================

    async def connect(self):

        try:

            await asyncio.to_thread(
                self._initialize_database
            )

            print(
                f"SQLite database ready: {self.db_path}"
            )

            return True

        except Exception as error:

            print(
                f"SQLite database initialization failed: {error}"
            )

            return False


    # ========================================================
    # CREATE USER
    # ========================================================

    async def create_user(
        self,
        username: str,
        password_hash: str,
        patient_name: Optional[str] = None,
        patient_id: Optional[str] = None,
        password_salt: Optional[str] = None
    ):

        try:

            return await asyncio.to_thread(
                self._create_user_sync,
                username,
                password_hash,
                patient_name,
                patient_id,
                password_salt
            )

        except Exception as error:

            print(
                f"SQLite create_user error: {error}"
            )

            return None


    def _create_user_sync(
        self,
        username: str,
        password_hash: str,
        patient_name: Optional[str],
        patient_id: Optional[str],
        password_salt: Optional[str]
    ):

        connection = self._get_connection()

        try:

            created_at = datetime.now(
                timezone.utc
            ).isoformat()


            cursor = connection.execute(
                """
                INSERT INTO users (
                    login_id,
                    password_hash,
                    password_salt,
                    patient_name,
                    patient_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username.strip(),
                    password_hash,
                    password_salt,
                    patient_name,
                    patient_id,
                    created_at
                )
            )


            connection.commit()


            return str(
                cursor.lastrowid
            )

        except sqlite3.IntegrityError:

            return None

        finally:

            connection.close()


    # ========================================================
    # GET USER BY USERNAME
    # ========================================================

    async def get_user_by_username(
        self,
        username: str
    ):

        try:

            return await asyncio.to_thread(
                self._get_user_by_username_sync,
                username
            )

        except Exception as error:

            print(
                f"SQLite get_user_by_username error: {error}"
            )

            return None


    def _get_user_by_username_sync(
        self,
        username: str
    ):

        connection = self._get_connection()

        try:

            row = connection.execute(
                """
                SELECT
                    id,
                    login_id,
                    password_hash,
                    password_salt,
                    patient_name,
                    patient_id,
                    created_at,
                    last_login
                FROM users
                WHERE login_id = ?
                """,
                (
                    username.strip(),
                )
            ).fetchone()


            if row is None:

                return None


            return dict(row)

        finally:

            connection.close()


    # ========================================================
    # GET USER BY ID
    # ========================================================

    async def get_user_by_id(
        self,
        user_id: str
    ):

        try:

            return await asyncio.to_thread(
                self._get_user_by_id_sync,
                user_id
            )

        except Exception as error:

            print(
                f"SQLite get_user_by_id error: {error}"
            )

            return None


    def _get_user_by_id_sync(
        self,
        user_id: str
    ):

        connection = self._get_connection()

        try:

            row = connection.execute(
                """
                SELECT
                    id,
                    login_id,
                    password_hash,
                    password_salt,
                    patient_name,
                    patient_id,
                    created_at,
                    last_login
                FROM users
                WHERE id = ?
                """,
                (
                    str(user_id),
                )
            ).fetchone()


            if row is None:

                return None


            return dict(row)

        finally:

            connection.close()


    # ========================================================
    # GET USER
    # ========================================================

    async def get_user(
        self,
        login_id: str
    ):

        return await self.get_user_by_username(
            login_id
        )


    # ========================================================
    # UPDATE LAST LOGIN
    # ========================================================

    async def update_last_login(
        self,
        user_id: str
    ):

        try:

            return await asyncio.to_thread(
                self._update_last_login_sync,
                user_id
            )

        except Exception as error:

            print(
                f"SQLite update_last_login error: {error}"
            )

            return False


    def _update_last_login_sync(
        self,
        user_id: str
    ):

        connection = self._get_connection()

        try:

            last_login = datetime.now(
                timezone.utc
            ).isoformat()


            connection.execute(
                """
                UPDATE users
                SET last_login = ?
                WHERE id = ?
                """,
                (
                    last_login,
                    str(user_id)
                )
            )


            connection.commit()

            return True

        finally:

            connection.close()


    # ========================================================
    # INSERT ANALYSIS
    # ========================================================

    async def insert_analysis(
        self,
        document: Dict[str, Any],
        user_id: Optional[str] = None,
        patient_information: Optional[Dict[str, Any]] = None
    ):

        try:

            return await asyncio.to_thread(
                self._insert_analysis_sync,
                document,
                user_id,
                patient_information
            )

        except Exception as error:

            print(
                f"SQLite insert_analysis error: {error}"
            )

            return None


    def _insert_analysis_sync(
        self,
        document: Dict[str, Any],
        user_id: Optional[str],
        patient_information: Optional[Dict[str, Any]]
    ):

        connection = self._get_connection()

        try:

            created_at = datetime.now(
                timezone.utc
            ).isoformat()


            # ------------------------------------------------
            # COPY DOCUMENT
            # ------------------------------------------------

            saved_document = dict(
                document
            )


            # ------------------------------------------------
            # PATIENT INFORMATION
            # ------------------------------------------------

            patient_information = (
                patient_information
                or saved_document.get(
                    "patient_information",
                    {}
                )
            )


            patient_case_id = (
                saved_document.get(
                    "patient_case_id"
                )
                or patient_information.get(
                    "case_id"
                )
            )


            patient_age = (
                saved_document.get(
                    "patient_age"
                )
                or patient_information.get(
                    "age"
                )
            )


            patient_gender = (
                saved_document.get(
                    "patient_gender"
                )
                or patient_information.get(
                    "gender"
                )
            )


            # ------------------------------------------------
            # DATABASE METADATA
            # ------------------------------------------------

            saved_document[
                "created_at"
            ] = created_at


            if user_id is not None:

                saved_document[
                    "user_id"
                ] = str(user_id)


            # ------------------------------------------------
            # JSON SERIALIZATION
            # ------------------------------------------------

            document_json = json.dumps(
                saved_document,
                default=str
            )


            # ------------------------------------------------
            # INSERT
            # ------------------------------------------------

            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    user_id,
                    patient_case_id,
                    patient_age,
                    patient_gender,
                    created_at,
                    document
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id)
                    if user_id is not None
                    else None,

                    patient_case_id,

                    patient_age,

                    patient_gender,

                    created_at,

                    document_json
                )
            )


            connection.commit()


            return str(
                cursor.lastrowid
            )

        finally:

            connection.close()


    # ========================================================
    # USER ANALYSES
    # ========================================================

    async def user_analyses(
        self,
        user_id: str,
        limit: int = 20
    ):

        try:

            return await asyncio.to_thread(
                self._user_analyses_sync,
                user_id,
                limit
            )

        except Exception as error:

            print(
                f"SQLite user_analyses error: {error}"
            )

            return []


    def _user_analyses_sync(
        self,
        user_id: str,
        limit: int
    ):

        connection = self._get_connection()

        try:

            limit = max(
                1,
                min(
                    int(limit),
                    200
                )
            )


            rows = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    patient_case_id,
                    patient_age,
                    patient_gender,
                    created_at,
                    document
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (
                    str(user_id),
                    limit
                )
            ).fetchall()


            results = []


            for row in rows:

                try:

                    document = json.loads(
                        row["document"]
                    )

                except Exception:

                    document = {}


                document["_id"] = str(
                    row["id"]
                )

                document["id"] = str(
                    row["id"]
                )

                document["created_at"] = (
                    row["created_at"]
                )


                if row["user_id"] is not None:

                    document["user_id"] = (
                        row["user_id"]
                    )


                if row["patient_case_id"] is not None:

                    document["patient_case_id"] = (
                        row["patient_case_id"]
                    )


                if row["patient_age"] is not None:

                    document["patient_age"] = (
                        row["patient_age"]
                    )


                if row["patient_gender"] is not None:

                    document["patient_gender"] = (
                        row["patient_gender"]
                    )


                results.append(
                    document
                )


            return results

        finally:

            connection.close()


    # ========================================================
    # RECENT ANALYSES
    # ========================================================

    async def recent_analyses(
        self,
        limit: int = 20,
        user_id: Optional[str] = None
    ):

        try:

            return await asyncio.to_thread(
                self._recent_analyses_sync,
                limit,
                user_id
            )

        except Exception as error:

            print(
                f"SQLite recent_analyses error: {error}"
            )

            return []


    def _recent_analyses_sync(
        self,
        limit: int,
        user_id: Optional[str]
    ):

        connection = self._get_connection()

        try:

            limit = max(
                1,
                min(
                    int(limit),
                    200
                )
            )


            if user_id is not None:

                rows = connection.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        patient_case_id,
                        patient_age,
                        patient_gender,
                        created_at,
                        document
                    FROM analyses
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (
                        str(user_id),
                        limit
                    )
                ).fetchall()

            else:

                rows = connection.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        patient_case_id,
                        patient_age,
                        patient_gender,
                        created_at,
                        document
                    FROM analyses
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (
                        limit,
                    )
                ).fetchall()


            results = []


            for row in rows:

                try:

                    document = json.loads(
                        row["document"]
                    )

                except Exception:

                    document = {}


                document["_id"] = str(
                    row["id"]
                )

                document["id"] = str(
                    row["id"]
                )

                document["created_at"] = (
                    row["created_at"]
                )


                if row["user_id"] is not None:

                    document["user_id"] = (
                        row["user_id"]
                    )


                if row["patient_case_id"] is not None:

                    document["patient_case_id"] = (
                        row["patient_case_id"]
                    )


                if row["patient_age"] is not None:

                    document["patient_age"] = (
                        row["patient_age"]
                    )


                if row["patient_gender"] is not None:

                    document["patient_gender"] = (
                        row["patient_gender"]
                    )


                results.append(
                    document
                )


            return results

        finally:

            connection.close()


    # ========================================================
    # DELETE USER ANALYSIS
    # ========================================================

    async def delete_analysis(
        self,
        analysis_id: str,
        user_id: str
    ):

        try:

            return await asyncio.to_thread(
                self._delete_analysis_sync,
                analysis_id,
                user_id
            )

        except Exception as error:

            print(
                f"SQLite delete_analysis error: {error}"
            )

            return False


    def _delete_analysis_sync(
        self,
        analysis_id: str,
        user_id: str
    ):

        connection = self._get_connection()

        try:

            connection.execute(
                """
                DELETE FROM reports
                WHERE analysis_id = ?
                """,
                (str(analysis_id),)
            )

            cursor = connection.execute(
                """
                DELETE FROM analyses
                WHERE id = ?
                AND user_id = ?
                """,
                (
                    str(analysis_id),
                    str(user_id)
                )
            )

            connection.commit()

            return cursor.rowcount > 0

        finally:

            connection.close()


    # ========================================================
    # DELETE ALL USER ANALYSES (CLEAR HISTORY)
    # ========================================================

    async def delete_all_user_analyses(
        self,
        user_id: str
    ):
        try:
            return await asyncio.to_thread(
                self._delete_all_user_analyses_sync,
                user_id
            )
        except Exception as error:
            print(f"SQLite delete_all_user_analyses error: {error}")
            return 0

    def _delete_all_user_analyses_sync(
        self,
        user_id: str
    ):
        connection = self._get_connection()
        try:
            connection.execute(
                """
                DELETE FROM reports
                WHERE user_id = ?
                   OR analysis_id IN (SELECT id FROM analyses WHERE user_id = ?)
                """,
                (str(user_id), str(user_id))
            )
            cursor = connection.execute(
                """
                DELETE FROM analyses
                WHERE user_id = ?
                """,
                (str(user_id),)
            )
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()


    # ========================================================
    # UPDATE ANALYSIS DOCUMENT
    # ========================================================

    async def update_analysis_document(
        self,
        analysis_id: str,
        document: Dict[str, Any]
    ):
        try:
            return await asyncio.to_thread(
                self._update_analysis_sync,
                analysis_id,
                document
            )
        except Exception as error:
            print(f"SQLite update_analysis_document error: {error}")
            return False

    def _update_analysis_sync(
        self,
        analysis_id: str,
        document: Dict[str, Any]
    ):
        connection = self._get_connection()
        try:
            doc_str = json.dumps(document)
            connection.execute(
                """
                UPDATE analyses
                SET document = ?
                WHERE id = ? OR document LIKE ?
                """,
                (
                    doc_str,
                    str(analysis_id),
                    f'%"{analysis_id}"%'
                )
            )
            connection.commit()
            return True
        finally:
            connection.close()

    # ========================================================
    # GET ANALYSIS BY ID
    # ========================================================

    async def get_analysis_by_id(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(
                self._get_analysis_by_id_sync,
                analysis_id
            )
        except Exception as error:
            print(f"SQLite get_analysis_by_id error: {error}")
            return None

    def _get_analysis_by_id_sync(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        connection = self._get_connection()
        try:
            # Query by integer id if numeric or inside document json
            row = connection.execute(
                """
                SELECT id, user_id, patient_case_id, patient_age, patient_gender, created_at, document
                FROM analyses
                WHERE id = ? OR document LIKE ?
                ORDER BY id DESC LIMIT 1
                """,
                (
                    str(analysis_id),
                    f'%"{analysis_id}"%'
                )
            ).fetchone()

            if not row:
                return None

            try:
                doc = json.loads(row["document"])
                if isinstance(doc, dict):
                    if "db_id" not in doc:
                        doc["db_id"] = row["id"]
                    return doc
                return None
            except Exception:
                return None
        finally:
            connection.close()


    # ========================================================
    # REPORT OPERATIONS
    # ========================================================

    async def insert_report(
        self,
        report_data: Dict[str, Any]
    ) -> Optional[str]:
        try:
            return await asyncio.to_thread(
                self._insert_report_sync,
                report_data
            )
        except Exception as error:
            print(f"SQLite insert_report error: {error}")
            return None

    def _insert_report_sync(
        self,
        report_data: Dict[str, Any]
    ) -> Optional[str]:
        connection = self._get_connection()
        try:
            report_id = str(report_data.get("id") or f"REP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
            analysis_id = str(report_data.get("analysis_id", ""))
            user_id = str(report_data.get("user_id", ""))
            patient_case_id = str(report_data.get("patient_case_id", ""))
            patient_id = str(report_data.get("patient_id", ""))
            findings = str(report_data.get("findings", ""))
            impression = str(report_data.get("impression", ""))
            report_version = int(report_data.get("report_version", 1))
            now_iso = datetime.now(timezone.utc).isoformat()
            created_at = str(report_data.get("created_at") or now_iso)
            updated_at = now_iso

            # Ensure id is in report_data dict
            report_data["id"] = report_id
            report_data["analysis_id"] = analysis_id
            report_data["created_at"] = created_at
            report_data["updated_at"] = updated_at
            report_content = json.dumps(report_data)

            connection.execute(
                """
                INSERT OR REPLACE INTO reports (
                    id,
                    analysis_id,
                    user_id,
                    patient_case_id,
                    patient_id,
                    findings,
                    impression,
                    report_version,
                    report_content,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    analysis_id,
                    user_id,
                    patient_case_id,
                    patient_id,
                    findings,
                    impression,
                    report_version,
                    report_content,
                    created_at,
                    updated_at
                )
            )
            connection.commit()
            return report_id
        finally:
            connection.close()


    async def get_report_by_id(
        self,
        report_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(
                self._get_report_by_id_sync,
                report_id
            )
        except Exception as error:
            print(f"SQLite get_report_by_id error: {error}")
            return None

    def _get_report_by_id_sync(
        self,
        report_id: str
    ) -> Optional[Dict[str, Any]]:
        connection = self._get_connection()
        try:
            row = connection.execute(
                """
                SELECT id, analysis_id, user_id, patient_case_id, patient_id,
                       findings, impression, report_version, report_content, created_at, updated_at
                FROM reports
                WHERE id = ?
                LIMIT 1
                """,
                (str(report_id),)
            ).fetchone()

            if not row:
                return None

            try:
                rep = json.loads(row["report_content"])
                return rep
            except Exception:
                return {
                    "id": row["id"],
                    "analysis_id": row["analysis_id"],
                    "findings": row["findings"],
                    "impression": row["impression"],
                    "report_version": row["report_version"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
        finally:
            connection.close()


    async def get_reports_for_analysis(
        self,
        analysis_id: str
    ) -> List[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(
                self._get_reports_for_analysis_sync,
                analysis_id
            )
        except Exception as error:
            print(f"SQLite get_reports_for_analysis error: {error}")
            return []

    def _get_reports_for_analysis_sync(
        self,
        analysis_id: str
    ) -> List[Dict[str, Any]]:
        connection = self._get_connection()
        try:
            rows = connection.execute(
                """
                SELECT id, analysis_id, user_id, patient_case_id, patient_id,
                       findings, impression, report_version, report_content, created_at, updated_at
                FROM reports
                WHERE analysis_id = ?
                ORDER BY created_at DESC
                """,
                (str(analysis_id),)
            ).fetchall()

            reports = []
            for r in rows:
                try:
                    rep = json.loads(r["report_content"])
                    reports.append(rep)
                except Exception:
                    reports.append({
                        "id": r["id"],
                        "analysis_id": r["analysis_id"],
                        "findings": r["findings"],
                        "impression": r["impression"],
                        "created_at": r["created_at"]
                    })
            return reports
        finally:
            connection.close()


    async def update_report(
        self,
        report_id: str,
        updated_data: Dict[str, Any]
    ) -> bool:
        try:
            return await asyncio.to_thread(
                self._update_report_sync,
                report_id,
                updated_data
            )
        except Exception as error:
            print(f"SQLite update_report error: {error}")
            return False

    def _update_report_sync(
        self,
        report_id: str,
        updated_data: Dict[str, Any]
    ) -> bool:
        connection = self._get_connection()
        try:
            # First fetch existing report to preserve unedited keys
            existing = self._get_report_by_id_sync(report_id)
            if not existing:
                return False

            merged = dict(existing)
            merged.update(updated_data)
            merged["updated_at"] = datetime.now(timezone.utc).isoformat()
            merged["report_version"] = int(existing.get("report_version", 1)) + 1

            findings = str(merged.get("findings", ""))
            impression = str(merged.get("impression", ""))
            content_str = json.dumps(merged)

            cursor = connection.execute(
                """
                UPDATE reports
                SET findings = ?,
                    impression = ?,
                    report_version = ?,
                    report_content = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    findings,
                    impression,
                    merged["report_version"],
                    content_str,
                    merged["updated_at"],
                    str(report_id)
                )
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()


    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self):

        return True