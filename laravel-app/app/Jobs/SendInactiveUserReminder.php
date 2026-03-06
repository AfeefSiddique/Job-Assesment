<?php

namespace App\Jobs;

use App\Models\ReminderLog;
use App\Models\User;
use Carbon\Carbon;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

class SendInactiveUserReminder implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    /**
     * The number of times the job may be attempted.
     */
    public int $tries = 3;

    /**
     * The number of seconds to wait before retrying.
     */
    public int $backoff = 60;

    public function __construct(
        public readonly User $user
    ) {}

    /**
     * Execute the job.
     */
    public function handle(): void
    {
        $now = Carbon::now();

        // Double-check: skip if the user was already reminded today
        // (guards against duplicate dispatches within the same day)
        $alreadySent = ReminderLog::where('user_id', $this->user->id)
            ->whereDate('sent_at', $now->toDateString())
            ->exists();

        if ($alreadySent) {
            Log::info("SendInactiveUserReminder: skipped (already reminded today)", [
                'user_id' => $this->user->id,
                'email'   => $this->user->email,
            ]);
            return;
        }

        // --- Simulate sending the reminder ---
        // Replace this block with a real Mailable / Notification when needed:
        //   Mail::to($this->user)->send(new InactivityReminderMail($this->user));
        //   $this->user->notify(new InactivityReminderNotification());

        $lastLogin = $this->user->last_login_at
            ? $this->user->last_login_at->toDateTimeString()
            : 'never';

        Log::channel('reminders')->info("REMINDER SENT", [
            'user_id'       => $this->user->id,
            'name'          => $this->user->name,
            'email'         => $this->user->email,
            'last_login_at' => $lastLogin,
            'sent_at'       => $now->toDateTimeString(),
        ]);

        // Record the reminder in the database
        ReminderLog::create([
            'user_id' => $this->user->id,
            'sent_at' => $now,
            'status'  => 'sent',
            'notes'   => "Reminder sent. Last login: {$lastLogin}",
        ]);
    }

    /**
     * Handle a job failure.
     */
    public function failed(\Throwable $exception): void
    {
        Log::error("SendInactiveUserReminder: job failed", [
            'user_id' => $this->user->id,
            'email'   => $this->user->email,
            'error'   => $exception->getMessage(),
        ]);

        // Optionally record the failure in the database
        ReminderLog::create([
            'user_id' => $this->user->id,
            'sent_at' => Carbon::now(),
            'status'  => 'failed',
            'notes'   => "Job failed: {$exception->getMessage()}",
        ]);
    }
}
