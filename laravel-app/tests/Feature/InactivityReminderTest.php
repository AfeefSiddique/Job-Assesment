<?php

namespace Tests\Feature;

use App\Jobs\SendInactiveUserReminder;
use App\Models\ReminderLog;
use App\Models\User;
use Carbon\Carbon;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Queue;
use Tests\TestCase;

class InactivityReminderTest extends TestCase
{
    use RefreshDatabase;

    /** @test */
    public function it_dispatches_jobs_for_inactive_users(): void
    {
        Queue::fake();

        // Active user — should NOT be reminded
        User::factory()->create(['last_login_at' => Carbon::now()]);

        // Inactive user — logged in 10 days ago
        $inactive = User::factory()->create([
            'last_login_at' => Carbon::now()->subDays(10),
        ]);

        // Never logged in — should also be reminded
        $neverLoggedIn = User::factory()->create(['last_login_at' => null]);

        $this->artisan('users:send-inactivity-reminders')->assertExitCode(0);

        Queue::assertPushed(SendInactiveUserReminder::class, 2);
        Queue::assertPushedOn('reminders', SendInactiveUserReminder::class);
    }

    /** @test */
    public function it_does_not_dispatch_job_if_user_reminded_today(): void
    {
        Queue::fake();

        $user = User::factory()->create([
            'last_login_at' => Carbon::now()->subDays(10),
        ]);

        // Simulate that a reminder was already sent today
        ReminderLog::create([
            'user_id' => $user->id,
            'sent_at' => Carbon::now(),
            'status'  => 'sent',
        ]);

        $this->artisan('users:send-inactivity-reminders')->assertExitCode(0);

        Queue::assertNothingPushed();
    }

    /** @test */
    public function job_records_reminder_log_on_execution(): void
    {
        $user = User::factory()->create([
            'last_login_at' => Carbon::now()->subDays(10),
        ]);

        (new SendInactiveUserReminder($user))->handle();

        $this->assertDatabaseHas('reminder_logs', [
            'user_id' => $user->id,
            'status'  => 'sent',
        ]);
    }

    /** @test */
    public function job_skips_user_already_reminded_today(): void
    {
        $user = User::factory()->create([
            'last_login_at' => Carbon::now()->subDays(10),
        ]);

        ReminderLog::create([
            'user_id' => $user->id,
            'sent_at' => Carbon::now(),
            'status'  => 'sent',
        ]);

        (new SendInactiveUserReminder($user))->handle();

        // Still only one log entry
        $this->assertCount(1, ReminderLog::where('user_id', $user->id)->get());
    }

    /** @test */
    public function dry_run_does_not_dispatch_jobs(): void
    {
        Queue::fake();

        User::factory()->create(['last_login_at' => Carbon::now()->subDays(10)]);

        $this->artisan('users:send-inactivity-reminders --dry-run')->assertExitCode(0);

        Queue::assertNothingPushed();
    }

    /** @test */
    public function custom_days_option_overrides_config(): void
    {
        Queue::fake();

        // Inactive for only 3 days — won't be picked up by the default 7-day window
        $user = User::factory()->create([
            'last_login_at' => Carbon::now()->subDays(3),
        ]);

        // With default (7 days): no jobs
        $this->artisan('users:send-inactivity-reminders')->assertExitCode(0);
        Queue::assertNothingPushed();

        // With --days=2: should dispatch
        $this->artisan('users:send-inactivity-reminders --days=2')->assertExitCode(0);
        Queue::assertPushed(SendInactiveUserReminder::class, 1);
    }
}
