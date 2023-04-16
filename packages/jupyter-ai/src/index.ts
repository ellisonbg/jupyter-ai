import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IWidgetTracker } from '@jupyterlab/apputils';
import { IDocumentWidget } from '@jupyterlab/docregistry';
import { IGlobalAwareness } from '@jupyterlab/collaboration';
import type { Awareness } from 'y-protocols/awareness';
import { buildChatSidebar } from './widgets/chat-sidebar';
import { SelectionWatcher } from './selection-watcher';
import { ChatHandler } from './chat_handler';

export type DocumentTracker = IWidgetTracker<IDocumentWidget>;

/**
 * Initialization data for the jupyter_ai extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter_ai:plugin',
  autoStart: true,
  optional: [IGlobalAwareness, ILayoutRestorer],
  activate: async (
    app: JupyterFrontEnd,
    globalAwareness: Awareness | null,
    restorer: ILayoutRestorer
  ) => {
    /**
     * Initialize selection watcher singleton
     */
    const selectionWatcher = new SelectionWatcher(app.shell);

    /**
     * Initialize chat handler, open WS connection
     */
    const chatHandler = new ChatHandler();
    await chatHandler.initialize();

    const chatWidget = buildChatSidebar(selectionWatcher, chatHandler, globalAwareness);

    /**
     * Add Chat widget to right sidebar
     */
    app.shell.add(
      chatWidget,
      'left', { rank: 2000 }
    );

    if (restorer) {
      restorer.add(chatWidget, 'jupyter-ai-chat');
    }
  }
};

export default plugin;
export type { InsertionContext } from './inserter';
