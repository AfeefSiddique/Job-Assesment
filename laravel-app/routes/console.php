<?php

use Illuminate\Support\Facades\Schedule;

/*
|--------------------------------------------------------------------------
| Scheduled Tasks
|--------------------------------------------------------------------------
|
| Register the inactivity-reminder command to run once a day at midnight.
| Laravel 11+ uses this file (routes/console.php) for scheduling.
|
*/

Schedule::command('users:send-inactivity-reminders')
    ->dailyAt('00:00')
    ->withoutOverlapping()           // Prevent concurrent runs
    ->runInBackground()              // Don't block other scheduled tasks
    ->appendOutputTo(storage_path('logs/scheduler.log'));
