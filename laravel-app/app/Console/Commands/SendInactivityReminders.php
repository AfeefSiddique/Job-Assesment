<?php

namespace App\Console\Commands;

use App\Jobs\SendInactiveUserReminder;
use App\Models\User;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Log;

class SendInactivityReminders extends Command
{
    /**
     * The name and signature of the console command.
     *
     * --days  : Override the inactivity threshold (in days)
     * --dry-run : List affected users without dispatching jobs
     */
    protected $signature = 'users:send-inactivity-reminders
                            {--days= : Override the inactivity period in days}
                            {--dry-run : Preview affected users without dispatching jobs}';

    protected $description = 'Find inactive users and dispatch reminder jobs for those not yet reminded today';

    public function handle(): int
    {
        $days    = $this->option('days')
            ? (int) $this->option('days')
            : config('inactive_users.inactivity_days', 7);
        $dryRun  = $this->option('dry-run');

        $this->info("Scanning for users inactive for more than {$days} day(s)…");

        $users = User::inactive($days)
            ->notRemindedToday()
            ->get();

        if ($users->isEmpty()) {
            $this->info('No inactive users found. Nothing to do.');
            Log::channel('reminders')->info('SendInactivityReminders: no inactive users found', [
                'inactivity_days' => $days,
            ]);
            return self::SUCCESS;
        }

        $this->info("Found {$users->count()} inactive user(s).");

        if ($dryRun) {
            $this->warn('[DRY-RUN] No jobs will be dispatched.');
            $this->table(
                ['ID', 'Name', 'Email', 'Last Login'],
                $users->map(fn (User $u) => [
                    $u->id,
                    $u->name,
                    $u->email,
                    $u->last_login_at?->toDateTimeString() ?? 'never',
                ])->toArray()
            );
            return self::SUCCESS;
        }

        $dispatched = 0;

        foreach ($users as $user) {
            SendInactiveUserReminder::dispatch($user)
                ->onQueue(config('inactive_users.queue', 'reminders'));

            $this->line("  → Queued reminder for: {$user->email}");
            $dispatched++;
        }

        $this->info("Done. Dispatched {$dispatched} job(s).");

        Log::channel('reminders')->info('SendInactivityReminders: jobs dispatched', [
            'inactivity_days' => $days,
            'dispatched'      => $dispatched,
        ]);

        return self::SUCCESS;
    }
}
