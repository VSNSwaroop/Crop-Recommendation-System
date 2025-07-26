import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    is_active = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create admin user if not exists
with app.app_context():
    db.create_all()
    admin = User.query.filter_by(email='admin@example.com').first()
    if not admin:
        admin = User(email='admin@example.com', password_hash=generate_password_hash('admin123'), is_active=True, is_admin=True)
        db.session.add(admin)
        db.session.commit()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))
        user = User(email=email, password_hash=generate_password_hash(password), is_active=False, is_admin=False)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Wait for admin activation.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid credentials.')
            return redirect(url_for('login'))
        if not user.is_active:
            flash('Account not activated. Wait for admin approval.')
            return redirect(url_for('login'))
        login_user(user)
        if user.is_admin:
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('farmer'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/farmer')
@login_required
def farmer():
    if not current_user.is_active or current_user.is_admin:
        return redirect(url_for('login'))
    return render_template('farmer.html', user=current_user)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for('login'))
    users = User.query.filter(User.email != 'admin@example.com').all()
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        user = User.query.get(user_id)
        if user:
            user.is_active = True
            db.session.commit()
            flash(f'Activated {user.email}')
        return redirect(url_for('admin'))
    return render_template('admin.html', users=users)

@app.route('/farmers')
@login_required
def farmers():
    if not current_user.is_admin:
        return redirect(url_for('login'))
    activated_users = User.query.filter(User.is_active == True, User.email != 'admin@example.com').all()
    not_activated_users = User.query.filter(User.is_active == False, User.email != 'admin@example.com').all()
    return render_template('farmers.html', activated_users=activated_users, not_activated_users=not_activated_users)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 