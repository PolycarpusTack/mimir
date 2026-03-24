// src/types/jira.ts — Phase 4 JIRA types
export type ApprovalStatus = 'Pending' | 'Pushing' | 'Pushed' | 'Failed' | 'Cancelled';

export interface ApprovalQueueItem {
  id: string;
  signal_id: string;
  status: ApprovalStatus;
  jira_project: string;
  jira_issue_key: string | null;
  ticket_title: string;
  ticket_body: string;
  spoke_label: string;
  domain_label: string;
  submitted_by: string;
  submitted_at: string;
  pushed_at: string | null;
  attempts: number;
  last_error: string | null;
}

export interface JiraPushRequest {
  signal_id: string;
  ticket_title: string;
  ticket_body: string;
  spoke_label: string;
  submitted_by: string;
}
