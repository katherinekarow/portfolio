-- Row-level security pattern: a public catalog + private per-user shelves.
-- Rewritten example of the approach used in a private project.

create table public.series (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  created_at timestamptz not null default now()
);

create table public.shelf_entries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  series_id uuid not null references public.series (id) on delete cascade,
  progress int not null default 0,
  unique (user_id, series_id)
);

alter table public.series enable row level security;
alter table public.shelf_entries enable row level security;

-- Everyone (even signed-out visitors) can browse the catalog.
create policy "catalog is readable by anyone"
  on public.series for select using (true);

-- Only admins (a custom JWT claim) can change catalog rows.
create policy "admins manage catalog"
  on public.series for all
  using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin')
  with check ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');

-- Users only ever see and change their own shelf.
create policy "own shelf only"
  on public.shelf_entries for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
