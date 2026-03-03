from db import db

class Reservation(db.Model):
    __tablename__ = "reservations"

    id = db.Column(db.Integer, primary_key=True)
    room = db.Column(db.String(20), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    hour = db.Column(db.String(20), nullable=False)

    leader_name = db.Column(db.String(50), nullable=False)
    leader_id = db.Column(db.String(50), nullable=False)
    leader_phone = db.Column(db.String(50), nullable=False)

    member_1_name = db.Column(db.String(50))
    member_1_id = db.Column(db.String(50))
    member_2_name = db.Column(db.String(50))
    member_2_id = db.Column(db.String(50))
    member_3_name = db.Column(db.String(50))
    member_3_id = db.Column(db.String(50))
    member_4_name = db.Column(db.String(50))
    member_4_id = db.Column(db.String(50))
    member_5_name = db.Column(db.String(50))
    member_5_id = db.Column(db.String(50))

    total_people = db.Column(db.Integer)
    duration = db.Column(db.Integer)

    # ✅ 인덱스 추가
    __table_args__ = (
        db.Index("ix_resv_room_date", "room", "date"),
        db.Index("ix_resv_leader_date", "leader_id", "date"),
    )


class PersonalReservation(db.Model):
    __tablename__ = "personal_reservations"

    id = db.Column(db.Integer, primary_key=True)
    seat = db.Column(db.String(10), nullable=False)  # 좌석 번호 1~7
    date = db.Column(db.String(20), nullable=False)
    hour = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.Integer, default=1)
    leader_name = db.Column(db.String(50), nullable=False)
    leader_id = db.Column(db.String(20), nullable=False)
    leader_phone = db.Column(db.String(20), nullable=False)
    total_people = db.Column(db.Integer, default=1)

    # ✅ 인덱스 추가
    __table_args__ = (
        db.Index("ix_pers_seat_date", "seat", "date"),
        db.Index("ix_pers_leader_date", "leader_id", "date"),
    )
