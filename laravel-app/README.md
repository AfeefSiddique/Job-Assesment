# Laravel Inactive User Reminder

A Laravel application that automatically detects inactive users and queues reminder jobs for each of them. Runs daily via a scheduled Artisan command.

---

## How It Works

1. A **scheduled command** (`users:send-inactivity-reminders`) runs every day at midnight.
2. It queries the `users` table for anyone whose `last_login_at` is older than the configured threshold (default: 7 days) **or** has never logged in.
3. Users who have already received a reminder **today** are skipped (deduplicated at both query and job level).
4. A `SendInactiveUserReminder` **queued job** is dispatched for each eligible user.
5. The job simulates sending a reminder and writes a record to the `reminder_logs` table (plus a dedicated log channel).

---

## Project Structure

```
app/
  Console/Commands/SendInactivityReminders.php  ← Artisan command + scheduler hook
  Jobs/SendInactiveUserReminder.php             ← Queued job (simulate/send reminder)
  Models/User.php                               ← scopeInactive + scopeNotRemindedToday
  Models/ReminderLog.php                        ← Tracks every sent reminder
  Http/Controllers/Auth/LoginController.php     ← Example: recording last_login_at

config/
  inactive_users.php                            ← Configurable inactivity_days & queue name

database/migrations/
  ..._add_last_login_at_to_users_table.php
  ..._create_reminder_logs_table.php

database/seeders/
  InactiveUserSeeder.php                        ← Test data

routes/
  console.php                                   ← Schedule::command() registration

tests/Feature/
  InactivityReminderTest.php                    ← Feature tests
```

---

## Setup Instructions

### Prerequisites

- PHP 8.2+
- Composer
- MySQL 8+
- A queue driver (database, Redis, etc.)

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/laravel-inactive-users.git
cd laravel-inactive-users

composer install
```

### 2. Environment configuration

```bash
cp .env.example .env
php artisan key:generate
```

Edit `.env` and set your database credentials:

```env
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=laravel_inactive_users
DB_USERNAME=root
DB_PASSWORD=secret

# Queue driver — use 'database' for zero extra infra, 'redis' for production
QUEUE_CONNECTION=database

# Optional overrides (defaults shown)
INACTIVE_USERS_DAYS=7
INACTIVE_USERS_QUEUE=reminders
```

### 3. Run migrations

```bash
php artisan migrate
```

### 4. (Optional) Seed test users

```bash
php artisan db:seed --class=InactiveUserSeeder
```

This creates four users:
| Name | Last Login | Will be reminded? |
|------|-----------|-------------------|
| Active User | Just now | No |
| Borderline User | 7 days ago | Yes |
| Inactive User | 10 days ago | Yes |
| Never Logged In | — (null) | Yes |

---

## Running the Scheduler

### Option A — Laravel's built-in scheduler (production)

Add a single cron entry on your server:

```cron
* * * * * cd /path-to-project && php artisan schedule:run >> /dev/null 2>&1
```

The scheduler will invoke `users:send-inactivity-reminders` automatically every day at **00:00**.

### Option B — Run the scheduler locally (development)

```bash
php artisan schedule:work
```

### Option C — Run the command directly (one-off / testing)

```bash
# Normal run
php artisan users:send-inactivity-reminders

# Override the inactivity threshold
php artisan users:send-inactivity-reminders --days=14

# Preview affected users without dispatching any jobs
php artisan users:send-inactivity-reminders --dry-run
```

---

## Running the Queue Worker

Start the worker to process dispatched reminder jobs:

```bash
# Process the 'reminders' queue
php artisan queue:work --queue=reminders

# With auto-restart on failure and memory limit (recommended for production)
php artisan queue:work --queue=reminders --tries=3 --backoff=60 --memory=128

# Using Supervisor (recommended for production)
# See: https://laravel.com/docs/queues#supervisor-configuration
```

> **Note:** If `QUEUE_CONNECTION=sync` is set in `.env`, jobs run immediately inline  
> (no worker needed) — convenient for local testing.

---

## Configuring the Inactivity Period

The threshold is configurable without touching any code:

| Method | How |
|--------|-----|
| `.env` | `INACTIVE_USERS_DAYS=14` |
| `config/inactive_users.php` | Change the default value |
| CLI flag | `php artisan users:send-inactivity-reminders --days=14` |

---

## Viewing Reminder Logs

**Database:**

```sql
SELECT r.id, u.name, u.email, r.sent_at, r.status, r.notes
FROM reminder_logs r
JOIN users u ON u.id = r.user_id
ORDER BY r.sent_at DESC;
```

**Log file** (`storage/logs/reminders.log`):

Each reminder job writes a structured entry like:

```
[2024-01-15 00:00:12] local.INFO: REMINDER SENT {"user_id":3,"name":"Inactive User","email":"inactive@example.com","last_login_at":"2024-01-05 10:30:00","sent_at":"2024-01-15 00:00:12"}
```

---

## Running Tests

```bash
php artisan test --filter InactivityReminderTest
```

Tests cover:
- Jobs are dispatched only for inactive users
- Users reminded today are skipped (no duplicate reminders)
- The job writes a `reminder_logs` record
- `--dry-run` flag dispatches nothing
- `--days` flag overrides the inactivity threshold

---

## Recording `last_login_at` on Login

Call `$user->recordLogin()` after a successful authentication. Example using an event listener (works with Breeze, Jetstream, Fortify):

```php
// app/Providers/AppServiceProvider.php

use Illuminate\Auth\Events\Login;
use Illuminate\Support\Facades\Event;

Event::listen(Login::class, function (Login $event) {
    $event->user->recordLogin();
});
```

Or call it directly in your `LoginController`:

```php
Auth::user()->recordLogin();
```

---

## Tech Stack

- **Laravel 11**
- **MySQL 8**
- **Laravel Queues** (database driver, easily swappable for Redis/SQS)
- **Laravel Scheduler** (single cron entry)
